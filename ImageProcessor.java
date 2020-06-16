import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

public class ImageProcessor {

    private static BufferedImage image = null;

    private static File file = null;

    /**
     * takes the image directory as String input
     *
     * @param inputImagePath an String
     */
    public static void inputImage(String inputImagePath) {
        try {
            file = new File(inputImagePath);
            image = ImageIO.read(file);
        } catch (IOException ex) {
            System.out.println("Failed to read image");
            System.out.println(ex);
        }
    }

    /**
     * resizes the image into new dimension
     *
     * @param inputImagePath
     * @param outputImagePath
     * @param scaledWidth
     * @param scaledHeight
     *
     */
    public static void resizeImage(String inputImagePath, String outputImagePath, int scaledWidth, int scaledHeight) {
        inputImage(inputImagePath);

        BufferedImage outputImage = new BufferedImage(scaledWidth,
                scaledHeight, image.getType());

        Graphics2D g = outputImage.createGraphics();
        g.drawImage(image, 0, 0, scaledWidth, scaledHeight, null);
        g.dispose();

        String formatName = outputImagePath.substring(outputImagePath.
                lastIndexOf(".") + 1);

        try {
            ImageIO.write(outputImage, formatName, new File(outputImagePath));
        } catch (IOException ex) {
            Logger.getLogger(ImageProcessor.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Couldn't resize the image.");
        }
    }

    /**
     * converts a set of input images to a matrix
     *
     * @param inputImagePath
     * @return
     */
    public static double[][] imageToMatrix(String[] inputImagePath) {
        int numOfImage = inputImagePath.length;
        String[] processedImagePath = new String[numOfImage];
        for (int i = 0; i < numOfImage; i++) {
            processedImagePath[i] = inputImagePath[i].substring(0, inputImagePath[i].lastIndexOf(".")) + "resized" + inputImagePath[i].substring(inputImagePath[i].lastIndexOf("."));
            resizeImage(inputImagePath[i], processedImagePath[i], 64, 64);
            rgbToGray(processedImagePath[i], processedImagePath[i]);
        }
        double[][] X = new double[64 * 64 * 3][numOfImage];
        for (int i = 0; i < numOfImage; i++) {
            inputImage(processedImagePath[i]);
            int height = image.getHeight();
            int width = image.getWidth();
            int prod = height * width;
            int j = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = image.getRGB(x, y);
                    X[j][i] = ((pixel >> 16) & 0xff) / 255;
                    X[prod + j][i] = ((pixel >> 8) & 0xff) / 255;
                    X[2 * prod + j][i] = (pixel & 0xff) / 255;
                    j++;
                }
            }
        }
        return X;
    }
    /**
     * Converts an rgb image to Grayscale, first parameter takes the input image location
     * and 2nd parameter takes the output image location, the function returns nothing.
     * 
     * @param in
     * @param out 
     */
    public static void rgbToGray(String in, String out) {
        BufferedImage img = null;
        File f = null;
        try {
            f = new File(in);
            img = ImageIO.read(f);
        } catch (IOException e) {
            System.out.println("Can't convert the image to Grayscale since no image found");
        }
        int width = img.getWidth();
        int height = img.getHeight();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int p = img.getRGB(x, y);
                int a = (p >> 24) & 0xff;
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                int avg = (r + g + b) / 3;
                p = (a << 24) | (avg << 16) | (avg << 8) | avg;
                img.setRGB(x, y, p);
            }
        }
        try {
            f = new File(out);
            ImageIO.write(img, "jpg", f);
        } catch (IOException e) {
            System.out.println("Converted the rgb image to gray but couldn't replace.");
        }
    }

}
