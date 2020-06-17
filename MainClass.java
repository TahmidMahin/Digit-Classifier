
public class MainClass {
    public static void main(String[] args) throws Exception {
        Dataset dt = new Dataset();
        dt.initialize(42000, 9, 1);
  
	NeuralNetwork nn = new NeuralNetwork(dt.getTrainingDataArray(), dt.getTrainingExpectedResultsArray());
	nn.gradientDescent(1000, 0.01);
		
		
		////////////////////////////////////////////////////////////
		// Enter your image directory here
		// Example: test = {"parent_directory/Umaar.jpg"}
		////////////////////////////////////////////////////////////
		String[] test = {};
		int[] result = nn.predict(dt.getTestDataArray());
		//for(int i=0;i<result.length;i++)
                    //System.out.println("");
    }
}
