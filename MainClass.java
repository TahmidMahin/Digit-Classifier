//sex
public class MainClass {
    public static void main(String[] args) throws Exception {
        Dataset dt = new Dataset();
//        dt.initialize(10000, 1, 4);
        dt.initialize(1800);

        NeuralNetwork nn = new NeuralNetwork(dt.getTrainingDataArray(), dt.getTrainingExpectedResultsArray());
        nn.gradientDescent(.001, 1);

		////////////////////////////////////////////////////////////
		// Enter your image directory here
		// Example: test = {"parent_directory/Umaar.jpg"}
		////////////////////////////////////////////////////////////
		String[] test = {};
		int[] result = nn.predict(dt.getTestDataArray());
		System.out.println(Dataset.calculateTestAccuracy(result));
		//for(int i=0;i<result.length;i++)
                    //System.out.println("");
    }
}
