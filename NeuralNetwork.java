
import java.util.HashMap;

public class NeuralNetwork {
	private int[] net = {784, 24, 12, 10};
	private double[][] input;
	private double[][] output;
	private int trainingExample;
	private HashMap<String, double[][]> dict = new HashMap<String, double[][]>();
	
	public NeuralNetwork() {
		for(int layer = 1; layer <= net.length; layer++) {
			dict.put("W" + Integer.toString(layer), Numpy.multiply(Numpy.randn(net[layer], net[layer-1]), 0.01));
			dict.put("b" + Integer.toString(layer), Numpy.zeros(net[layer], 1));
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
	
	public double propagate() {
		double[][] W1 = dict.get("W1");
		double[][] b1 = dict.get("b1");
		double[][] W2 = dict.get("W2");
		double[][] b2 = dict.get("b2");
		double[][] W3 = dict.get("W3");
		double[][] b3 = dict.get("b3");
		
		double[][] Z1 = Numpy.add(Numpy.dot(W1, input), b1);
		double[][] A1 = Numpy.tanh(Z1);
		double[][] Z2 = Numpy.add(Numpy.dot(W2, A1), b2);
		double[][] A2 = Numpy.relu(Z2);
		double[][] Z3 = Numpy.add(Numpy.dot(W3, A2), b3);
		double[][] A3 = Numpy.softmax(Z3);
		
		double cost = Numpy.sum(Numpy.multiply(output, Numpy.log(A3)));
		cost /= -trainingExample;
		return cost;
	}
}
