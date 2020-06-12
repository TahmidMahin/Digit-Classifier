
import java.util.Arrays;
import java.util.Random;

public class Numpy {

    Numpy() {
    }
    /**
     * initializes the given matrix with Gaussian values
     * @param mat
     * @return a new matrix
     */
    public static double[][] initialize(double[][] mat) {
        Random rand = new Random();
        double[][] W = new double[mat.length][mat[0].length];
        for (double[] W1 : W) {
            for (int j = 0; j < W[0].length; j++) {
                W1[j] = rand.nextGaussian();
            }
        }
        return W;
    }
    /**
     * initializes the given matrix with 2nd parameter value
     * @param mat
     * @param k
     * @return a new matrix
     */
    public static double[][] initialize(double[][] mat,int k) {
        double[][] W = new double[mat.length][mat[0].length];
        for (double[] W1 : W) {
            for (int j = 0; j < W[0].length; j++) {
                W1[j] = k;
            }
        }
        return W;
    }
    /**
     * initializes the given matrix with random double values [0,1)
     * @param mat
     * @return a new matrix
     */
    public static double[][] initializeRand(double[][] mat) {
        Random rand = new Random();
        double[][] W = new double[mat.length][mat[0].length];
        for (double[] W1 : W) {
            for (int j = 0; j < W[0].length; j++) {
                W1[j] = rand.nextDouble();
            }
        }
        return W;
    }
    /////////logarithm ,exponential & other functions///////////////

    /**
     * Transposes a given matrix
     *
     * @param mat
     * @return
     */
    public static double[][] transpose(double[][] mat) {
        double[][] result = new double[mat[0].length][mat.length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                result[i][j] = mat[j][i];
            }
        }
        return result;
    }

    /**
     * finds the element wise exponential value of a given matrix
     *
     * @param mat
     * @return
     */
    public static double[][] exp(double[][] mat) {
        double[][] result = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                result[i][j] = Math.exp(mat[i][j]);
            }
        }
        return result;
    }

    /**
     * finds the element wise natural logarithm value of a given matrix
     *
     * @param mat
     * @return
     */
    public static double[][] log(double[][] mat) {
        double[][] result = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                result[i][j] = Math.log(mat[i][j]);
            }
        }
        return result;
    }

    /**
     * finds the SIGMOID value of a given matrix
     *
     * @param mat
     * @return
     */
    public static double[][] sigmoid(double[][] mat) {
        return inverse(add(1, exp(subtract(0, mat))));
    }

    /**
     * finds the element wise tanh value of a given matrix, and returns the
     * matrix but doesn't change the original one
     *
     * @param mat
     * @return temp
     */
    public static double[][] tanh(double[][] mat) {
        double[][] temp = exp(multiply(-2, mat));
        return multiply(subtract(1, temp), inverse(add(1, temp)));
    }

    public static double[][] relu(double[][] mat) {
        double[][] result = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                result[i][j] = Math.max(0, mat[i][j]);
            }
        }
        return result;
    }

    /**
     * finds the element wise inverse value of a given matrix
     *
     * @param mat
     * @return
     */
    public static double[][] inverse(double[][] mat) {
        double[][] temp = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                temp[i][j] = 1 / mat[i][j];
            }
        }
        return temp;
    }
    //////////////////////////////////////////////////////////////////

    /////////////////broadcasting///////////////////
    /**
     * resizes a vector to a matrix
     *
     * @param a
     * @param row
     * @param col
     * @return
     */
    public static double[][] broadcast(double[][] a, int row, int col) {
        double[][] b = new double[row][col];
        int prev_row = a.length, prev_col = a[0].length;
        if (prev_row == row && prev_col == col) {
            return a;
        }
        if (prev_row == 1) {
            for (int i = 0; i < col; i++) {
                double temp = a[0][i];
                for (int j = 0; j < row; j++) {
                    b[j][i] = temp;
                }
            }
        } else if (prev_col == 1) {
            for (int i = 0; i < row; i++) {
                double temp = a[i][0];
                Arrays.fill(b[i], temp);
            }
        }
        return b;
    }

    ///////////////element wise multiplication operations////////////////
    /**
     * finds the element wise multiplication of two given matrix
     *
     * @param mat1
     * @param mat2
     * @return
     */
    public static double[][] multiply(double[][] mat1, double[][] mat2) {
        int row1 = mat1.length;
        int col1 = mat1[0].length;
        int row2 = mat2.length;
        int col2 = mat2[0].length;
        if (row1 != row2 && col1 != col2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        if (row1 != row2) {
            if (row1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }
        if (col1 != col2) {
            if (col1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }

        double[][] result = new double[row1][col1];
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
        return result;
    }

    /**
     * finds the element wise multiplication of a given value and a given matrix
     *
     * @param mat1
     * @param num
     * @return
     */
    public static double[][] multiply(double[][] mat1, double num) {
        double[][] mat2 = new double[mat1.length][mat1[0].length];
        for (double[] mat21 : mat2) {
            Arrays.fill(mat21, num);
        }
        return multiply(mat1, mat2);
    }

    /**
     * finds the element wise multiplication of a given value and a given matrix
     *
     * @param num
     * @param mat1
     * @return
     */
    public static double[][] multiply(double num, double[][] mat1) {
        double[][] mat2 = new double[mat1.length][mat1[0].length];
        for (double[] mat21 : mat2) {
            Arrays.fill(mat21, num);
        }
        return multiply(mat1, mat2);
    }

    /////////////////////////////////////////////////////////
    /////////Various addition operation//////////////////////
    /**
     * finds the element wise addition of two given matrix
     *
     * @param mat1
     * @param mat2
     * @return
     */
    public static double[][] add(double[][] mat1, double[][] mat2) {
        int row1 = mat1.length;
        int col1 = mat1[0].length;
        int row2 = mat2.length;
        int col2 = mat2[0].length;
        if (row1 != row2 && col1 != col2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        if (row1 != row2) {
            if (row1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }
        if (col1 != col2) {
            if (col1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }
        double[][] result = new double[row1][col1];
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    /**
     * finds the element wise addition of a given value and a given matrix
     *
     * @param mat1
     * @param num
     * @return
     */
    public static double[][] add(double[][] mat1, double num) {
        double[][] mat2 = new double[mat1.length][mat1[0].length];
        for (double[] mat21 : mat2) {
            Arrays.fill(mat21, num);
        }
        return add(mat1, mat2);
    }

    /**
     * finds the element wise addition of a given value and a given matrix
     *
     * @param num
     * @param mat1
     * @return
     */
    public static double[][] add(double num, double[][] mat1) {
        double[][] mat2 = new double[mat1.length][mat1[0].length];
        for (double[] mat21 : mat2) {
            Arrays.fill(mat21, num);
        }
        return add(mat1, mat2);
    }

    /**
     * finds the row wise summation of a given matrix
     *
     * @param mat
     * @return
     */
    public static double sum(double[][] mat) {
        double result = 0.0;
        for (double[] mat1 : mat) {
            for (int j = 0; j < mat[0].length; j++) {
                result += mat1[j];
            }
        }
        return result;
    }

    /**
     * finds the summation of a double array
     *
     * @param mat
     * @return
     */
    public static double sum(double[] mat) {
        double result = 0.0;
        for (int i = 0; i < mat.length; i++) {
            result += mat[i];
        }
        return result;
    }
    //////////////////////////////////////////////////////////

    //////////////////////various subtract operations/////////////////
    /**
     * finds the element wise subtraction of a given matrix & a value
     *
     * @param mat1
     * @param num
     * @return
     */
    public static double[][] subtract(double[][] mat1, double num) {
        return add(mat1, -num);
    }

    /**
     * finds the element wise subtraction of a given matrix & a value
     *
     * @param num
     * @param mat1
     * @return
     */
    public static double[][] subtract(double num, double[][] mat1) {
        double[][] temp = new double[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++) {
            for (int j = 0; j < mat1[0].length; j++) {
                temp[i][j] = -mat1[i][j];
            }
        }
        return add(temp, num);
    }

    /**
     * finds the element wise subtraction of two given matrix
     *
     * @param mat1
     * @param mat2
     * @return
     */
    public static double[][] subtract(double[][] mat1, double[][] mat2) {
        int row1 = mat1.length;
        int col1 = mat1[0].length;
        int row2 = mat2.length;
        int col2 = mat2[0].length;
        if (row1 != row2 && col1 != col2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        if (row1 != row2) {
            if (row1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }
        if (col1 != col2) {
            if (col1 == 1) {
                mat1 = broadcast(mat1, row2, col2);
                row1 = row2;
                col1 = col2;
            } else {
                mat2 = broadcast(mat2, row1, col1);
                row2 = row1;
                col2 = col1;
            }
        }
        double[][] result = new double[row1][col1];
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    }

    ////////////////////////////////////////////////////////////////
    /**
     * finds the dot products of two matrix
     *
     * @param mat1
     * @param mat2
     * @return
     */
    public static double[][] dot(double[][] mat1, double[][] mat2) {
        int m1 = mat1.length;
        int n1 = mat1[0].length;
        int m2 = mat2.length;
        int n2 = mat2[0].length;
        if (n1 != m2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n1; k++) {
                    c[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return c;
    }

    /**
     * prints a matrix
     *
     * @param mat
     */
    public static void print(double[][] mat) {
        System.out.print("[");
        for (double[] row : mat) {
            System.out.print("[");
            for (double column : row) {
                System.out.print(column + "    ");
            }
            System.out.println("]");
        }
        System.out.println("]");
    }

}
