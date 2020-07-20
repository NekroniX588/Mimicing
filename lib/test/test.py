from utils.nms_wrapper import nms
import numpy as np
import pickle
import sys
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.timer import Timer

class Tester():
	def __init__(self, cfg):
		self.cfg = cfg

	def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
		model.eval()

		dataset = data_loader.dataset
		num_images = len(dataset)
		num_classes = detector.num_classes
		all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
		empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

		_t = Timer()

		for i in iter(range((num_images))):
			img = dataset.pull_image(i)
			scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
			if use_gpu:
				images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
			else:
				images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

			_t.tic()
			# forward
			out = model(images, phase='eval')

			# detect
			detections = detector.forward(out)

			time = _t.toc()

			# TODO: make it smart:
			for j in range(1, num_classes):
				cls_dets = list()
				for det in detections[0][j]:
					if det[0] > 0:
						d = det.cpu().numpy()
						score, box = d[0], d[1:]
						box *= scale
						box = np.append(box, score)
						cls_dets.append(box)
				if len(cls_dets) == 0:
					cls_dets = empty_array
				all_boxes[j][i] = np.array(cls_dets)

			# log per iter
			log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
					prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
					time=time)
			sys.stdout.write(log)
			sys.stdout.flush()

		# write result to pkl
		with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
			pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

		# currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
		print('Evaluating detections')
		ap,mean_ap = data_loader.dataset.evaluate_detections(all_boxes, output_dir)
		writer.add_scalar('mAP', mean_ap, epoch)
		model.train()



	def nms_process(self, num_classes, i, scores, boxes, min_thresh, all_boxes, max_per_image):
		for j in range(1, num_classes):  # ignore the bg(category_id=0)
			inds = np.where(scores[:, j] > min_thresh)[0]
			if len(inds) == 0:
				all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
				continue
			c_bboxes = boxes[inds]
			c_scores = scores[inds, j]
			c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
				np.float32, copy=False)

			soft_nms = False
			keep = nms(c_dets, self.cfg.IOU_THRESHOLD , force_cpu=soft_nms)
			# keep only the highest boxes
			keep = keep[:self.cfg.MAX_DETECTIONS]
			c_dets = c_dets[keep, :]
			all_boxes[j][i] = c_dets
		if max_per_image > 0:
			image_scores = np.hstack([all_boxes[j][i][:, -1]
									  for j in range(1, num_classes)])
			if len(image_scores) > max_per_image:
				image_thresh = np.sort(image_scores)[-max_per_image]
				for j in range(1, num_classes):
					keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
					all_boxes[j][i] = all_boxes[j][i][keep, :]

	def test_fast_nms(self, model, data_loader, detector, output_dir, use_gpu, writer, epoch):
		model.eval()

		dataset = data_loader.dataset
		num_images = len(dataset)
		num_classes = detector.num_classes
		all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
		empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

		_t = Timer()

		for i in iter(range((num_images))):
			img = dataset.pull_image(i)
			scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
			with torch.no_grad():
				if use_gpu:
					images = dataset.preproc(img)[0].unsqueeze(0).cuda()
				else:
					images = dataset.preproc(img)[0].unsqueeze(0)

			_t.tic()
			# forward
			out = model(images, phase='eval')
			# detect
			boxes, scores = detector.forward(out)
			boxes = (boxes[0] * scale).cpu().numpy()
			scores = scores[0].cpu().numpy()
			self.nms_process(num_classes, i, scores, boxes,
					 self.cfg.SCORE_THRESHOLD , all_boxes, self.cfg.MAX_DETECTIONS)

			time = _t.toc()

			# log per iter
			log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
					prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
					time=time)
			sys.stdout.write(log)
			sys.stdout.flush()

		# write result to pkl
		with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
			pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
		# with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
		#     all_boxes = pickle.load(f)

		# currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
		print('Evaluating detections')
		ap,mean_ap = data_loader.dataset.evaluate_detections(all_boxes, output_dir)
		writer.add_scalar('mAP', mean_ap, epoch)
		model.train()