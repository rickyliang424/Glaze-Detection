#Authored by Ricky Liang (8/20/24)

import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt

class GlazeDetection:
    def __init__(self, pr=0.75, th=100, order=5, size=100, stride=20, lowcut=1, highcut=25):
        """
        Function to detect glaze defects of an image or a video frame
        Parameters:
            img_gray: grayscale image or video frame (ndarray)
            pr: percentile of all ROI intensity scores to show (%)
            th: pixel value threshold (0~255)
            order: order of butterworth filter
            size: ROI window size (pixel)
            stride: movement of the kernel (pixel)
            lowcut: low cutoff frequency of bandpass filter (Hz) (> 0)
            highcut: high cutoff frequency of bandpass filter (Hz) (< size/2)
        Returns:
            ROI_score_arr: contain all ROI average scores
                            col0: y position of the upper-left corner of the ROI
                            col1: x position of the upper-left corner of the ROI
                            col2: average score of the ROI
            heatmap: 2D array with average intensity scores for each pixel
        """
        self.pr = pr
        self.th = th
        self.order = order
        self.size = size
        self.stride = stride
        self.lowcut = lowcut
        self.highcut = highcut
        self.camera = None
        self.glaze_processing = False
        self.is_glaze_paused = False
    
    def butter_bandpass_filter(self, data):
        normal_lowcut = self.lowcut / (0.5 * self.size)
        normal_highcut = self.highcut / (0.5 * self.size)
        b, a = butter(self.order, [normal_lowcut, normal_highcut], btype='band')
        y = filtfilt(b, a, data)
        return y
    
    def compute_ROI_score(self, ROI):
        N = ROI.shape[0]
        xf = fftfreq(N, 1/self.size)[:N//2]
        # filter out noise using two param "lowcut" and "highcut"
        ROI_filt = self.butter_bandpass_filter(ROI.T).T
        # apply sobel filter to sharpen edges (y-direction, kernel_size=3)
        ROI_filt = cv2.Sobel(ROI_filt, cv2.CV_64F, 0, 1, ksize=3)
        # set thresholds to eliminate background and black lines
        ROI_filt[ROI_filt > self.th] = self.th
        ROI_filt[ROI_filt < -self.th] = -self.th
        # take fft and find peak frequency and its amplitude
        ROI_fft = np.abs(fft(ROI_filt, axis=0)[:N//2])
        peak_amp = 2/N * np.max(ROI_fft, axis=0)
        peak_freq = xf[np.argmax(ROI_fft, axis=0)]
        # compute average intensity scores
        scores = peak_freq * peak_amp
        score_avg = np.mean(scores)
        return score_avg

    def process_ROIs(self, img_gray):
        ROI_score_list = []
        heatmap = np.zeros(img_gray.shape)
        heatmap_cnt = np.ones(img_gray.shape)
        
        # set sliding windows with pre-defined stride and ROI size
        for h in range(img_gray.shape[0] % self.stride // 2, img_gray.shape[0] - self.size, self.stride):
            for w in range(img_gray.shape[1] % self.stride // 2, img_gray.shape[1] - self.size, self.stride):
                crop_u = h
                crop_l = w
                crop_d = crop_u + self.size
                crop_r = crop_l + self.size
                ROI = img_gray[crop_u:crop_d, crop_l:crop_r]
                score_avg = self.compute_ROI_score(ROI)
                ROI_score_list.append([h, w, score_avg])
                # record ROI scores for each sliding widnow
                heatmap[crop_u:crop_d, crop_l:crop_r] += score_avg
                heatmap_cnt[crop_u:crop_d, crop_l:crop_r] += 1
        
        ROI_score_arr = np.array(ROI_score_list, dtype=int)
        heatmap = heatmap / heatmap_cnt
        return ROI_score_arr, heatmap
