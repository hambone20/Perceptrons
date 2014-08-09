package perceptrons

import io.Source
import collection.mutable.ArrayBuffer
import util.Random


object BinaryPerceptrons {
	var tolerance: Double = 0.1
	var learningRate: Double = 0.05
	var maxEpochs = 15 // max epochs we want to run
	var doEpochs = 0 // shortcuts to run a set amount of epochs
}

class BinaryPerceptrons (val dataFile: String = "optdigits", 
                         var learningRate: Double = 0.2, var tolerance: Double = 0.001, 
                         val numOutput: Int = 1, val numBias: Int = 1){
	// construct
    // Setup companion object
    BinaryPerceptrons.tolerance = tolerance
    BinaryPerceptrons.learningRate = learningRate
    
	private val dataNamesLines = Source.fromFile(dataFile + ".names").getLines
	
	private val header = dataNamesLines.next()
	
	// CREATE nodes
  private val delims = Array(' ', ',', '.')
	private val nodesStr = header.mkString.split(delims).filter(x => x != "")
	var nodes = nodesStr.map(x => x.toInt)
	
	// CREATE attributes
	var attributes = ArrayBuffer[Attribute]() // holds all attributes
	var numAttributes = 0
	for(nextLine <- dataNamesLines){
	  numAttributes += 1
	  attributes += new Attribute(nextLine.takeWhile(_ != ':'), -1.0 + Random.nextDouble() + Random.nextDouble())
	}
	
	// CREATE biases
	var biases = ArrayBuffer[Attribute]()
	for(i <- 0 until numBias) biases += new Attribute("bias" + i, -1.0 + Random.nextDouble() + Random.nextDouble(), 1) 
	
	// CREATE tests
	var tests = ArrayBuffer[(Int, Int)]()
	for(x <- nodes)for(y <- nodes) if(!tests.contains(Tuple2(y,x)) && x != y)tests += Tuple2(x, y)

	val trainingData = getData(dataFile + ".train") // training set
	val testData = getData(dataFile + ".test") // test set
	
	var perceptrons = ArrayBuffer[Perceptron]()
	createPerceptrons() // populate all tests

	//METHODS	
	
	def createPerceptrons() {
		for((x, y) <- tests){
		  perceptrons += new Perceptron(x, y, attributes, biases)
		}
	}
	
	def getData(file: String) = {
	  val data = Source.fromFile(file).getLines
	  var mapData = collection.mutable.Map[Int, ArrayBuffer[Array[Int]]]()
	  for(x <- data){  
		val inst = x.mkString.split(delims).toArray.map(y => y.toInt)
		var arr: ArrayBuffer[Array[Int]] = mapData.getOrElse(inst(numAttributes), ArrayBuffer[Array[Int]]())

		var app: Array[Int] = inst.take(numAttributes)
		arr += (app)
		mapData += (inst(numAttributes) -> arr)
	  }
	  mapData
	}
	
	def doTrain() = {
	  var trainSubjects = perceptrons

	  while(trainSubjects.size > 0){
	    println(s"Running training epoch with ${trainSubjects.size}")
	    
		  for(per <- trainSubjects){
		    val curEpoch = per.runEpoch(trainingData, true)
		  }
	    trainSubjects = perceptrons.filter(_.epochData.epochsDone() == false)
	  }
	}
	
	def doTest() = {
	  for(per <- perceptrons){
		  	  val testEpoch = per.runEpoch(testData, false)
		  	  per.testNum = testEpoch._2
		  	  per.testCorrect = testEpoch._1
		  }
	  
	}
	
	def doMultiClassTest() = {
	  println("running multi class test")
	  var matrix = ArrayBuffer[(Int, Int, Double, Perceptron)]() // output, target, confidence, perceptron
	  var maxConfidence: Double = 0
	  var maxConfidencePer: Perceptron = null
	  var maxConfidenceOutput: Int = 10
	  var num = 0
	  for((target, data) <- testData){
	    for(example <- data){
	      num += 1
		    for(per <- perceptrons){
		      per.doExample(target, example, false)
		    	  
		      var confidence = Math.abs(per.epochSum)
		      if(confidence > maxConfidence || maxConfidence == 0){
		        maxConfidence = confidence
		        maxConfidencePer = per
		        maxConfidenceOutput = per.output()
		      }
		    }
	    	matrix += Tuple4(maxConfidenceOutput, target, maxConfidence, maxConfidencePer)
	    	maxConfidence = 0
	    	maxConfidencePer = null
	    }
	  }
	  println(s"Num instances = $num")
	  matrix
	}
}