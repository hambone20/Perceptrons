import perceptrons._

object PerpTest extends App {
	println("Perceptrons")
	val perceptrons = new BinaryPerceptrons("optdigits")
	BinaryPerceptrons.tolerance = 0.00001
	BinaryPerceptrons.learningRate = 0.05
    if(this.args.size > 0){
      BinaryPerceptrons.learningRate = this.args(0).toDouble
    }
	if(BinaryPerceptrons.doEpochs == 0)
		println(s"Tolerance is: ${BinaryPerceptrons.tolerance}")
	else
	  println(s"Running $BinaryPerceptrons.doEpochs epochs")
	  
	perceptrons.doTrain()

	var totalAccuracy = 0.0
	var totalSize = 0
	var totalEpochs = 0
	
	println(s"Training Data (Learning Rate = ${BinaryPerceptrons.learningRate})")
	for(per <- perceptrons.perceptrons){
	    val accuracy = per.getAccuracy()
	    totalAccuracy += accuracy
	    totalSize += 1
	    totalEpochs += per.numEpochs
		println(f"Perceptron $per: Epochs = ${per.numEpochs} Accuracy = ${accuracy * 100}%2.1f%%")
	}
	println(f"Average accuracy on training set was ${totalAccuracy / totalSize * 100}%2.1f%%")

	println(f"Average number of epochs was ${totalEpochs.toDouble / perceptrons.perceptrons.size}%2.1f")
	
	perceptrons.doTest()
	totalAccuracy = 0.0
	totalSize = 0
	println(s"\nTest Data (Learning Rate = ${BinaryPerceptrons.learningRate})")
	for(per <- perceptrons.perceptrons){
	  val accuracy = per.getTestAccuracy()
	  totalAccuracy += accuracy
	  totalSize += 1
	  println(f"Perceptron $per: Accuracy = ${accuracy * 100}%2.1f%% Correct = ${per.testCorrect} Incorrect = ${per.testNum-per.testCorrect}")
	}
	println(f"Average accuracy on testing set was ${totalAccuracy / totalSize * 100}%2.1f%%")
	
	/*val matrix = perceptrons.doMultiClassTest()
	for((output, target, maxConfidence, per) <- matrix){
	  println(s"output = $output, target = $target with $maxConfidence")
	  println(per)
	}*/
}