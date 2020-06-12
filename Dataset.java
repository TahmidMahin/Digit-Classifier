import java.util.*;
import java.io.*;

/*
    a class that deals with datasets. it has the following operations:
        NOTE:
            - This class is designed to be used as a static class. Which means,
            NO OBJECT OF THIS CLASS SHOULD BE CREATED. ALL methods of this
            class should be called like this,
                Dataset.methodName(...);
            - all array indexing starts from 1.


        public static void initialize(double a, double b):
            - THE USER MUST CALL THIS CLASS BEFORE CALLING ANYTHING ELSE.
            - initializes (loads the training images and test images from
            files into arrays)

            - What are a and b?
                We have a total of 42000 labeled images. We want to divide
                these images into two parts. One part, for training, another
                part, for testing.

                Now, (a : b) = (number_of_training_images : number_of_test_images).
                Where, number_of_training_images + number_of_test_images = 42000.

                So, if there were 10 images, and a = 4 and b = 1, then 8 images
                will be used for training and 2 will be used for testing. Getter
                methods (getNoOfTrainingImages() and getNoOfTestImages()) (see below)
                have been created for expressing how many images are being used for each
                task. If a <= 0 or b <= 0, then the function will terminate without doing
                anything.


        public static int[][] getTrainingDataArray():
            - returns the 2D matrix of Training images.
            - The returned array has a dimension of [43000] * [800].

            Each column represents the pixels of a particular training example
            (a particular image).
            Each image consists of 784 pixels.

                Let
                    int array[][] = Dataset.getTrainingDataArray();

                So, array[i][1] to array[i][784] represents the greyscale values
                of all the pixels of the i'th training example.

             NOTE: As we saw in the initialize() method, 42000 images are getting
             divided into two parts. So not every column represent an image. To know
             the exact number of images being used for training, use the method
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
            - similar to the nethod calculateTrainingAccuracy(int[] in),
            but this one calculates the accuracy of the testing examples
            instead of the training examples.


        public static int getNoOfTrainingImages()
            - returns the number of images used for training the NN.


        public static int getNoOfTestImages()
            - returns the number of images used for testing the NN's accuracy.


        public static int getNoOfPixelsPerImage()
            - returns the number of pixels that each image consists (784).


        public static int[] getTrainingExpectedResultsArray():
            - returns an array whose i'th index represents the number that
            the i'th training image shows.


        public static int[] getTestExpectedResultsArray()
            - returns an array whose i'th index represents the number that
            the i'th test image should show.
 */

public class Dataset {
    private static int nOfLabeledImages = 42000;
    private static int nOfPixelsPerImage = 784;
    private static double ratioOfTrainingData = (double) 8;
    private static double ratioOfTestData = (double) 2;
    private static int nOfTrainingImages;
    private static int nOfTestImages;

    private static String imagesPath = "datasets/images.txt";
    private static String expectedResultsPath = "datasets/expected results.txt";

    private static int[][] trainingDataArray = new int[43000][800];
    private static int[] trainingExpectedResultsArray = new int[43000];
    private static int[][] testDataArray = new int[43000][800];
    private static int[] testExpectedResultsArray = new int[43000];

    public static void initialize(double a, double b) throws Exception {
        if(a <= 0.0 || b <= 0.0) return;
        ratioOfTrainingData = a;
        ratioOfTestData = b;

        nOfTrainingImages = (int) (nOfLabeledImages * (ratioOfTrainingData) / (ratioOfTestData + ratioOfTrainingData));
        nOfTestImages = 42000 - nOfTrainingImages;

//        System.out.println(nOfTrainingImages);
//        System.out.println(nOfTestImages);

        File file1 = new File(imagesPath);
        Scanner scImage = new Scanner(file1);
        File file2 = new File(expectedResultsPath);
        Scanner scExp = new Scanner(file2);

        int i = 0, lc1, lc2;

        for(lc1 = 1; lc1 <= nOfTrainingImages; lc1++) {
            for (lc2 = 1; lc2 <= nOfPixelsPerImage; lc2++) {
                trainingDataArray[lc1][lc2] = Integer.parseInt(scImage.nextLine());
//                if (lc1 == nOfTrainingImages) System.out.print(trainingDataArray[lc1][lc2] + " ");
            }
            trainingExpectedResultsArray[lc1] = Integer.parseInt(scExp.nextLine());
        }

        for(lc1 = nOfTrainingImages + 1; lc1 <= nOfLabeledImages; lc1++) {
            for(lc2 = 1; lc2 <= nOfPixelsPerImage; lc2++) {
                testDataArray[lc1 - nOfTrainingImages][lc2] = Integer.parseInt(scImage.nextLine());
                if (lc1 == nOfLabeledImages) System.out.print(testDataArray[lc1 - nOfTrainingImages][lc2] + " ");
            }
            testExpectedResultsArray[lc1 - nOfTrainingImages] = Integer.parseInt(scExp.nextLine());
        }

//        if(!scImage.hasNextLine()) System.out.println("done");
//        if(!scExp.hasNextLine()) System.out.println("done");
    }

    public static double calculateTrainingAccuracy(int[] in) {
        int lg, ok;
        for(lg = 1, ok = 0; lg <= nOfTrainingImages; lg++)
            if(in[lg] == trainingExpectedResultsArray[lg]) ok++;
        return ok * 100.0 / nOfTrainingImages;
    }
    public static double calculateTestAccuracy(int[] in) {
        int lg, ok;
        for(lg = 1, ok = 0; lg <= nOfTestImages; lg++)
            if(in[lg] == testExpectedResultsArray[lg]) ok++;
        return ok * 100.0 / nOfTestImages;
    }

    public static int[][] getTrainingDataArray() {
        return trainingDataArray;
    }
    public static int[][] getTestDataArray() {
        return testDataArray;
    }
    public static int[] getTrainingExpectedResultsArray() {
        return trainingExpectedResultsArray;
    }
    public static int[] getTestExpectedResultsArray() {
        return testExpectedResultsArray;
    }
    public static int getNoOfTrainingImages() {
        return nOfTrainingImages;
    }
    public static int getNoOfTestImages() {
        return nOfTestImages;
    }
    public static int getNoOfPixelsPerImage() {
        return nOfPixelsPerImage;
    }
}