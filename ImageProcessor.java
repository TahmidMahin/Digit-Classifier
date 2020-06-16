
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageProcessor {

    private static BufferedImage image = null;

    private static File file = null;

    /**
     * takes the image directory as String input
     *
     * @param inputImagePath an String777
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
     * @throws IOException
     */
    public static void resizeImage(String inputImagePath, String outputImagePath, int scaledWidth, int scaledHeight) throws IOException {
        inputImage(inputImagePath);

        BufferedImage outputImage = new BufferedImage(scaledWidth,
                scaledHeight, image.getType());

        Graphics2D g = outputImage.createGraphics();
        g.drawImage(image, 0, 0, scaledWidth, scaledHeight, null);
        g.dispose();

        String formatName = outputImagePath.substring(outputImagePath.
                lastIndexOf(".") + 1);

        ImageIO.write(outputImage, formatName, new File(outputImagePath));
    }

    /**
     * converts a set of input images to a matrix
     *
     * @param inputImagePath
     * @return
     * @throws IOException
     */
    public static double[][] imageToMatrix(String[] inputImagePath, int scaledWidth, int scaledHeight) throws IOException {
        int numOfImage = inputImagePath.length;
        String[] processedImagePath = new String[numOfImage];
        for (int i = 0; i < numOfImage; i++) {
            processedImagePath[i] = inputImagePath[i].substring(0, inputImagePath[i].lastIndexOf(".")) + "resized" + inputImagePath[i].substring(inputImagePath[i].lastIndexOf("."));
            resizeImage(inputImagePath[i], processedImagePath[i], scaledWidth, scaledHeight);
        }
        double[][] X = new double[scaledWidth * scaledWidth][numOfImage];
        for (int ithImage = 0; ithImage < numOfImage; ithImage++) {
            inputImage(processedImagePath[ithImage]);
            int height = image.getHeight();
            int width = image.getWidth();
            for (int y = 0, pixel = 0; y < height; y++, pixel++) {
                for (int x = 0; x < width; x++) {
                    int p = image.getRGB(x, y);
                    double r = (double)((p >> 16) & 0xff);
                    double g = (double)((p >> 8) & 0xff);
                    double b = (double)(p & 0xff);
                    double avg = (r+g+b) / 3;
                    X[pixel][ithImage] = avg;
                }
            }
        }
        return X;
    }
}
