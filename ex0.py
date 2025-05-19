import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
            image_path (str): Path to the input image.
            colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")


        # If I had more images, I would do this:
        # images: list = [cv2.imread(os.path.join(self._image_directory, i), cv2.IMREAD_COLOR) for i in os.listdir(self._image_directory)]

        self._colour_type: str = colour_type

        # Loading colour image with BGR encoding.
        # Output shape: (H, W, 3)
        self._image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if colour_type == ("RGB"):
            # Transform image from BGR encoding to RGB encoding (basically reversing the third dimension)
            self._image = self._image[:,:,::-1] # alternative to the cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif colour_type == "Gray":
            # Transform image from BGR encoding to grayscale encoding
            self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def get_image_data(self) -> tuple[np.ndarray, str]:
        """
        Return the image data (image and colour scheme).

        Returns:
            tuple(np.ndarray, str): Loaded image and current colour scheme.
        """
        return self._image, self._colour_type

    def show_image(self, title: str = "Image"):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # By using matplotlib.pyplt.imshow, the image is shown perfectly
        # By using the opencv.imshow, the colours are not aqequate. This is because in the constructor I transformed the
        # image encoding from BGR to RGB. So by changing it back with the convert_colour function, it should appear right again
        # And yes, my theory was proven rigth.


        # Show the image depending on the colour type.
        if self._colour_type in ["RGB", "BGR"]:
            plt.imshow(self._image)
        else:
            plt.imshow(self._image, cmap="gray")

        plt.title(label=title)
        plt.axis("off")
        plt.show()

        """
        self.convert_colour()
        cv2.imshow('Image', self._image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """


    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
            image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.

        total_image_path: str = os.path.join(self._image_directory, image_title)
        """
        plt.imsave(total_image_path, self._image)
        """

        cv2.imwrite(filename=total_image_path, img=self._image)


    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # So as it can be seen RGB and BGR store the colour channels in reverse order.
        # So we just need to reverse the third dimension of the image, as the image is stored
        # in this way: (H, W, COLOUR_CHANNELS)
        self._image = self._image[:, :, ::-1]
        self._colour_type = "BGR" if self._colour_type == "RGB" else "RGB"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
            clip_min (int): Minimum image colour intensity.
            clip_max (int): Maximum image colour intensity.
        """

        # The colour intensity is the value that is stored in each colour channel
        # Set all pixel values below clip_min to clip_min and above clip_max to clip_max
        self._image[self._image < clip_min] = clip_min
        self._image[self._image > clip_max] = clip_max

    def flip_image(self, flip_value: int = 0):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
            flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # [row, column, colour_channel]

        if flip_value == 0:
            # Vertical flip means every column needs to be reversed
            self._image = self._image[::-1, :, :]

        if flip_value == 1:
            # Horizontal flip means every row needs to be reversed
            self._image = self._image[:, ::-1, :]

        if flip_value == 2:
            # Do horizontal and vertical flip as well
            self._image = self._image[::-1, ::-1, :]



if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="RGB")

    # Testing get_image_data
    img, colour_type = processor.get_image_data()
    print(f'Image: {img}')
    print(f'Image shape: {img.shape}')
    print(f'Colour type: {colour_type}')

    # Testing show_image
    processor.show_image()

    # Testing flip image
    processor.flip_image(flip_value=0)
    processor.show_image(title="Vertical flip")
    processor.flip_image(flip_value=0)

    processor.flip_image(flip_value=1)
    processor.show_image(title="Horizontal flip")
    processor.flip_image(flip_value=1)

    processor.flip_image(flip_value=2)
    processor.show_image(title="Horizontal and vertical flip")
    processor.flip_image(flip_value=2)


    # Testing clip image
    processor.clip_image(150,200)
    processor.show_image(title="Clipped image")

    # Testing save image
    processor.save_image(image_title="saved_image.png")


