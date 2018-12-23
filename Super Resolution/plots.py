import matplotlib.pyplot as plt
import numpy as np
from processImage import denormalize_HR


class plots(object):

    def __init__(self, LRDim, HRDim, LRimages, HRimages, store_path):
        self.HRDim = HRDim
        self.LRDim = LRDim
        self.store_path = store_path
        self.LRimages = LRimages
        self.HRimages = HRimages

    def plot_g_d_loss(self, loss):
        plt.plot(loss["d_loss"], 'r-x', label="Discriminator Loss")
        plt.plot(loss["g_loss"], 'b-x', label="Generator Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_generated_images(self, index, generator, step):
        test_input = self.LRimages[index]
        ground_truth = self.HRimages[index]

        test_input = test_input.reshape(1, self.LRDim, self.LRDim, 3)
        # ground_truth = ground_truth.reshape(1, HRDim, HRDim, 3)

        prediction = generator.predict(test_input)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], denormalize_HR(ground_truth), denormalize_HR(prediction[0])]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i])
            plt.axis('off')
        plt.savefig(self.store_path + '\\gan_generated_image_step_%d.png' % step)
        plt.clf()
        # plt.show()
