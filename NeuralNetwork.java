import java.util.HashMap;
 
public class NeuralNetwork {
    private int[] net = {784, 24, 12, 10};
    private double[][] input;
    private double[][] output;
    private int trainingExample;
    private HashMap<String, double[][]> parameters = new HashMap<String, double[][]>();
    private HashMap<String, double[][]> cache = new HashMap<String, double[][]>();
    private HashMap<String, double[][]> grads = new HashMap<String, double[][]>();
   
    public NeuralNetwork() {
        for(int layer = 1; layer <= net.length; layer++) {
            parameters.put("W" + Integer.toString(layer), Numpy.multiply(Numpy.randn(net[layer], net[layer-1]), 0.01));
            parameters.put("b" + Integer.toString(layer), Numpy.zeros(net[layer], 1));
        }
    }
   
    public NeuralNetwork(double[][] input, double[][] output) {
        this();
        this.input = input;
        this.output = output;
        trainingExample = input[0].length;
    }
   
    public NeuralNetwork(double[][] input, int[] labels) {
        this();
        this.input = input;
        output = new double[10][labels.length];
        for(int j=0; j<labels.length; j++)
            output[labels[j]][j] = 1.0;
        trainingExample = labels.length;
    }
   
    private double[][] reluBackward(double[][] mat) {
        double[][] result = new double[mat.length][mat[0].length];
        for(int i=0; i<mat.length; i++)
            for(int j=0; j<mat[0].length; j++)
                result[i][j] = (mat[i][j] < 0.0 ? 0.0 : 1.0);
        return result;
    }
   
    private double[][] tanhBackward(double[][] mat) {
        double[][] temp = Numpy.tanh(mat);
        return Numpy.subtract(1, Numpy.multiply(temp, temp));
    }
   
    public double[][] forwardPropagation(double[][] X) {
        double[][] W1 = parameters.get("W1");
        double[][] b1 = parameters.get("b1");
        double[][] W2 = parameters.get("W2");
        double[][] b2 = parameters.get("b2");
        double[][] W3 = parameters.get("W3");
        double[][] b3 = parameters.get("b3");
       
        double[][] Z1 = Numpy.add(Numpy.dot(W1, X), b1);
        double[][] A1 = Numpy.tanh(Z1);
        double[][] Z2 = Numpy.add(Numpy.dot(W2, A1), b2);
        double[][] A2 = Numpy.relu(Z2);
        double[][] Z3 = Numpy.add(Numpy.dot(W3, A2), b3);
        double[][] A3 = Numpy.softmax(Z3);
       
        cache.put("Z1", Z1);
        cache.put("A1", A1);
        cache.put("Z2", Z2);
        cache.put("A2", A2);
        cache.put("Z3", Z3);
        cache.put("A3", A3);
       
        return A3;
    }
   
    public double computeCost(double[][] yhat) {
        double cost = Numpy.sum(Numpy.multiply(output, Numpy.log(yhat)));
        cost /= -trainingExample;
        return cost;
    }
   
    public void backwardPropagation() {
        double[][] W2 = parameters.get("W2");
        double[][] W3 = parameters.get("W3");
       
        double[][] Z1 = cache.get("Z1");
        double[][] A1 = cache.get("A1");
        double[][] Z2 = cache.get("Z2");
        double[][] A2 = cache.get("A2");
        double[][] A3 = cache.get("A3");
       
        double[][] dZ3 = Numpy.subtract(A3, output);
        double[][] dW3 = Numpy.multiply(Numpy.dot(dZ3, Numpy.transpose(A2)), 1/trainingExample);
        double[][] db3 = Numpy.multiply(Numpy.sum(dZ3, 0), 1/trainingExample);
        double[][] dZ2 = Numpy.multiply(Numpy.dot(Numpy.transpose(W3), dZ3), reluBackward(Z2));
        double[][] dW2 = Numpy.multiply(Numpy.dot(dZ2, Numpy.transpose(A1)), 1/trainingExample);
        double[][] db2 = Numpy.multiply(Numpy.sum(dZ2, 0), 1/trainingExample);
        double[][] dZ1 = Numpy.multiply(Numpy.dot(Numpy.transpose(W2), dZ2), tanhBackward(Z1));
        double[][] dW1 = Numpy.multiply(Numpy.dot(dZ1, Numpy.transpose(input)), 1/trainingExample);
        double[][] db1 = Numpy.multiply(Numpy.sum(dZ1, 0), 1/trainingExample);
       
        grads.put("dW1", dW1);
        grads.put("db1", db1);
        grads.put("dW2", dW2);
        grads.put("db2", db2);
        grads.put("dW3", dW3);
        grads.put("db3", db3);
    }
   
    public void updateParameters(double learningRate) {
        double[][] W1 = parameters.get("W1");
        double[][] b1 = parameters.get("b1");
        double[][] W2 = parameters.get("W2");
        double[][] b2 = parameters.get("b2");
        double[][] W3 = parameters.get("W3");
        double[][] b3 = parameters.get("b3");
       
        double[][] dW1 = grads.get("dW1");
        double[][] db1 = grads.get("db1");
        double[][] dW2 = grads.get("dW2");
        double[][] db2 = grads.get("db2");
        double[][] dW3 = grads.get("dW3");
        double[][] db3 = grads.get("db3");
       
        W1 = Numpy.subtract(W1, Numpy.multiply(learningRate, dW1));
        b1 = Numpy.subtract(b1, Numpy.multiply(learningRate, db1));
        W2 = Numpy.subtract(W1, Numpy.multiply(learningRate, dW2));
        b2 = Numpy.subtract(b1, Numpy.multiply(learningRate, db2));
        W3 = Numpy.subtract(W1, Numpy.multiply(learningRate, dW3));
        b3 = Numpy.subtract(b1, Numpy.multiply(learningRate, db3));
       
        parameters.put("W1", W1);
        parameters.put("b1", b1);
        parameters.put("W2", W2);
        parameters.put("b2", b2);
        parameters.put("W3", W3);
        parameters.put("b3", b3);
    }
   
    public void gradientDescent(int iterations, double learningRate) {
        for(int itr=0; itr<=iterations; itr++) {
            double[][] yhat = forwardPropagation(input);
            double cost = computeCost(yhat);
            backwardPropagation();
            updateParameters(learningRate);
            if(itr%100 == 0)
                System.out.println("cost after iteration " + itr + ": " +cost);
        }
    }
   
    public double[][] predict(double[][] testData) {
        return forwardPropagation(testData);
    }
   
}
