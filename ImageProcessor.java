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
    
    private static int scaledImageWidth = 28;
    private static int scaledImageHeight = 28;
    

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
            resizeImage(inputImagePath[i], processedImagePath[i], scaledImageWidth, scaledImageHeight);
        }
        double[][] X = new double[scaledImageWidth * scaledImageHeight][numOfImage];
        for (int i = 0; i < numOfImage; i++) {
            inputImage(processedImagePath[i]);
            int height = image.getHeight();
            int width = image.getWidth();
            for (int y = 0, j = 0; y < height; y++, j++) {
                for (int x = 0; x < width; x++) {
                    int pixel = image.getRGB(x, y);
                    double r = ((pixel >> 16) & 0xff) / 255;
                    double g = ((pixel >> 8) & 0xff) / 255;
                    double b = (pixel & 0xff) / 255;
                    double avg = (r+g+b) / 3;
                    X[j][i] = avg;
                }
            }
        }
        return X;
    }

}
