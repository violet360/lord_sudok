import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator

def show_image(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	img = in_img.copy()

	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	for point in points:
		img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
	show_image(img)
	return img



def plot_many_images(images, titles, rows=1, columns=2):
	for i, image in enumerate(images):
		plt.subplot(rows, columns, i+1)
		plt.imshow(image, 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

def pre_process_image(img, skip_dilate=False):
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	proc = cv2.bitwise_not(proc, proc)
	kernal = np.ones((2,2), np.uint8)
	# proc = cv2.erode(proc, kernal, iterations=1)
	proc = cv2.dilate(proc, kernal, iterations = 1)
	return proc

img = cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE)
processed = pre_process_image(img)

def find_corners_of_largest_polygon(img):
	contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	polygon = contours[0]
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



def distance_between(p1, p2):
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	m = cv2.getPerspectiveTransform(src, dst)
	img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	return cv2.warpPerspective(img, m, (int(side), int(side)))


img = cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE)
processed = pre_process_image(img)
corners = find_corners_of_largest_polygon(processed)
# show_image(processed)
cropped = crop_and_warp('sudoku.jpg', corners)
show_image(cropped)
