
import java.util.HashMap;

public class NeuralNetwork {
	private int[] net = {784, 24, 12, 10};
	private double[][] input;
	private double[][] output;
	private int trainingExample;
	private HashMap<String, double[][]> parameters = new HashMap<String, double[][]>();
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
	
	public double propagate() {
		double[][] W1 = parameters.get("W1");
		double[][] b1 = parameters.get("b1");
		double[][] W2 = parameters.get("W2");
		double[][] b2 = parameters.get("b2");
		double[][] W3 = parameters.get("W3");
		double[][] b3 = parameters.get("b3");
		
		double[][] Z1 = Numpy.add(Numpy.dot(W1, input), b1);
		double[][] A1 = Numpy.tanh(Z1);
		double[][] Z2 = Numpy.add(Numpy.dot(W2, A1), b2);
		double[][] A2 = Numpy.relu(Z2);
		double[][] Z3 = Numpy.add(Numpy.dot(W3, A2), b3);
		double[][] A3 = Numpy.softmax(Z3);
		
		double cost = Numpy.sum(Numpy.multiply(output, Numpy.log(A3)));
		cost /= -trainingExample;
		
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
		
		return cost;
	}
}
