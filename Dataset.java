import java.util.*;
import java.io.*;

/*
    a class that deals with datasets. it has the following operations:
        NOTE:
            - This class is designed to be used as a static class. Which means,
            NO OBJECT OF THIS CLASS SHOULD BE CREATED. ALL methods of this
            class should be called like this,
                Dataset.methodName(...);
            - all array indexing starts from 0.


        public static void initialize(int _nOfUsedImages, double a, double b):
            - THE USER MUST CALL THIS METHOD BEFORE CALLING ANYTHING ELSE.
            - initializes (loads the training images and test images from
            files into matrices)

            - _nOfUsedImages:
                We have a total of 42000 labeled images. This parameter should
                state how many of these images are going to be used in total for
                both training and testing. Limit: 0 < _nOfUsedImages <= 42000.

            - What are a and b?
                We want to divide _nOfUsedImages images into two parts. One part,
                for training, another part, for testing.

                Now, (a : b) = (number_of_training_images : number_of_test_images).
                Where, number_of_training_images + number_of_test_images = _nOfUsedImages.

                So, if there were 10 images, and a = 4 and b = 1, then 8 images
                would be used for training and 2 would be used for testing. Getter
                methods (getNoOfTrainingImages() and getNoOfTestImages()) (see below)
                have been created for expressing how many images are being used for each
                task. If a <= 0 or b <= 0, then the method will terminate without doing
                anything.


        public static int[][] getTrainingDataArray():
            - returns the 2D matrix of Training images.
            - The returned array has a dimension of [42000] * [784].

            Each row represents the pixels of a particular training example
            (a particular image).
            Each image consists of 784 pixels.

                Let
                    int array[][] = Dataset.getTrainingDataArray();

                So, array[i][1] to array[i][784] represents the greyscale values
                of all the pixels of the i'th training example.

             NOTE: As we saw in the initialize() method, _nOfUsedImages images are
             getting divided into two parts. So not every row represent an image.
             To know the exact number of images being used for training, use the method
             getNoOfTrainingImages() (see below).


        public static int[][] getTestDataArray():
            - similar to the method getTrainingDataArray(), but this one
            returns the 2D matrix of Testing images instead of the Training images.

            Use method getNoOfTestImages() to know the exact number of images
            being used for testing.


        public static double calculateTrainingAccuracy(int[] in):
            - given an array containing the NN's output for all the training
            examples, this method calculates the accuracy of the NN in percentage.
            - parameter: an array whose i'th index represents the neural
            network's output for the i'th training example. array indexing
            of the parameter array should start from 1.


        public static int[] calculateTestAccuracy():
            - similar to the method calculateTrainingAccuracy(int[] in),
            but this one calculates the accuracy of the testing examples
            instead of the training examples.


        public static int[] getTrainingExpectedResultsArray():
            - returns an array whose i'th index represents the number that
            the i'th training image shows.


        public static int[] getTestExpectedResultsArray()
            - returns an array whose i'th index represents the number that
            the i'th test image should show.


        public static int getNoOfPixelsPerImage()
            - returns the number of pixels that each image consists (784).


        public static int getNoOfUsedImages()
            - returns the total number of images used for training
            and testing the NN.


        public static int getNoOfTrainingImages()
            - returns the number of images used for training the NN.


        public static int getNoOfTestImages()
            - returns the number of images used for testing the NN's accuracy.
 */

public class Dataset {
    private static int nOfLabeledImages = 42000;
    private static int nOfUsedImages;
    private static int nOfPixelsPerImage = 784;
    private static double ratioOfTrainingData = (double) 8;
    private static double ratioOfTestData = (double) 2;
    private static int nOfTrainingImages;
    private static int nOfTestImages;

    private static String imagesPath = "images.txt";
    private static String expectedResultsPath = "expected results.txt";

    private static double[][] trainingDataArray;
    private static int[] trainingExpectedResultsArray;
    private static double[][] testDataArray;
    private static int[] testExpectedResultsArray;

    public static void initialize(int _nOfUsedImages, double a, double b) throws Exception {
        if(a <= 0.0 || b <= 0.0 || _nOfUsedImages <= 0) return;

        nOfUsedImages = Math.min(_nOfUsedImages, nOfLabeledImages);
        ratioOfTrainingData = a;
        ratioOfTestData = b;

        nOfTrainingImages = (int) (nOfUsedImages * (ratioOfTrainingData) / (ratioOfTestData + ratioOfTrainingData));
        nOfTestImages = nOfUsedImages - nOfTrainingImages;

        System.out.println(nOfTrainingImages);
        System.out.println(nOfTestImages);

        trainingDataArray = new double[nOfTrainingImages][nOfPixelsPerImage];
        testDataArray = new double[nOfTestImages][nOfPixelsPerImage];
        trainingExpectedResultsArray = new int[nOfTrainingImages];
        testExpectedResultsArray = new int[nOfTestImages];

        File file1 = new File(imagesPath);
        Scanner scImage = new Scanner(file1);
        File file2 = new File(expectedResultsPath);
        Scanner scExp = new Scanner(file2);

        int i = 0, lc1, lc2;

        for(lc1 = 0; lc1 < nOfTrainingImages; lc1++) {
            for (lc2 = 0; lc2 < nOfPixelsPerImage; lc2++) {
                trainingDataArray[lc1][lc2] = (double) Integer.parseInt(scImage.nextLine());
//                if (lc1 == nOfTrainingImages) System.out.print(trainingDataArray[lc1][lc2] + " ");
            }
            trainingExpectedResultsArray[lc1] = Integer.parseInt(scExp.nextLine());
        }

        for(lc1 = nOfTrainingImages; lc1 < nOfUsedImages; lc1++) {
            for(lc2 = 0; lc2 < nOfPixelsPerImage; lc2++) {
                testDataArray[lc1 - nOfTrainingImages][lc2] = (double) Integer.parseInt(scImage.nextLine());
//                if (lc1 == nOfLabeledImages) System.out.print(testDataArray[lc1 - nOfTrainingImages][lc2] + " ");
            }
            testExpectedResultsArray[lc1 - nOfTrainingImages] = Integer.parseInt(scExp.nextLine());
        }

        trainingDataArray = Numpy.multiply(trainingDataArray, 1/255.0);
        testDataArray = Numpy.multiply(testDataArray, 1/255.0);

//        if(!scImage.hasNextLine()) System.out.println("done");
//        if(!scExp.hasNextLine()) System.out.println("done");
    }

    public static double[][] getTrainingDataArray() {
        return Numpy.transpose(trainingDataArray);
    }
    public static double[][] getTestDataArray() {
        return Numpy.transpose(testDataArray);
    }

    public static double calculateTrainingAccuracy(double[] in) {
        int lg, ok;
        for(lg = 0, ok = 0; lg < nOfTrainingImages; lg++)
            if(in[lg] == trainingExpectedResultsArray[lg]) ok++;
        return ok * 100.0 / nOfTrainingImages;
    }
    public static double calculateTestAccuracy(double[] in) {
        int lg, ok;
        for(lg = 0, ok = 0; lg < nOfTestImages; lg++)
            if(in[lg] == testExpectedResultsArray[lg]) ok++;
        return ok * 100.0 / nOfTestImages;
    }

    public static int[] getTrainingExpectedResultsArray() {
        return trainingExpectedResultsArray;
    }
    public static int[] getTestExpectedResultsArray() {
        return testExpectedResultsArray;
    }
    public static int getNoOfPixelsPerImage() {
        return nOfPixelsPerImage;
    }
    public static int getNoOfUsedImages() {
        return nOfUsedImages;
    }
    public static int getNoOfTrainingImages() {
        return nOfTrainingImages;
    }
    public static int getNoOfTestImages() {
        return nOfTestImages;
    }
}
