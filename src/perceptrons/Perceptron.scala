package perceptrons

import collection.mutable.ArrayBuffer
import util.Random

class Perceptron (var pos: Int, var neg: Int, var attributes: ArrayBuffer[Attribute], 
                  biases: ArrayBuffer[Attribute]){
  val epochData = new Epoch
  var epochSum = 0.0
  
  var testNum = 0
  var testCorrect = 0
  
  var maxDelta = BinaryPerceptrons.tolerance + 1
  
  def output() = if(Math.signum(epochSum) >= 0) pos else neg
  
  def doExample(targetRef: Int, example: Array[Int], doDelta: Boolean = true) = {// returns output
    epochSum = 0.0
    
    for((x, y) <- attributes zip example){ // load inputs
      x.input = y
    }
    
    for(x <- biases ++ attributes){ // calculate 
      epochSum += x.calc()
    }
    
    val output = Math.signum(epochSum) // perform output
        
    if (doDelta){ // delta w_i = learning_rate(diff) * input where diff = target - output
      val target = if(targetRef == pos) 1 else -1 // normalize target
      val diff = target - output
      
      for(x <- attributes ++ biases){
        val delta = BinaryPerceptrons.learningRate * diff * x.input
        if(Math.abs(delta) > maxDelta)
          maxDelta = delta
          
        // change weights
        x.weight += delta
      }
    }
    output
  }
  
def addEpoch(a: Int, b: Int) { epochData.addEpoch(a, b) }
  
def runEpoch(data: collection.mutable.Map[Int, ArrayBuffer[Array[Int]]], train: Boolean ) = {
	  var correct, num = 0
	  val randomizedData = Random.shuffle(data.filter(_._1 == pos) ++ data.filter(_._1 == neg))
	  for(x <- randomizedData){
	    for(y <- x._2){
	      num += 1
	      val output = doExample(x._1, y, train) // do example
	      if ((output >= 0 && x._1 == pos) || (output < 0 && x._1 == neg))
	        correct += 1
	  	}
	  }
	  
	  if(train)
	    addEpoch(correct, num)
	  
  (correct, num)
}

  def getAccuracy() : Double = {
    epochData.getAccuracy()
  }
  
  def getTestAccuracy() : Double = {
    if (testNum == 0) return 0.0
    else return testCorrect.toDouble / testNum
  }
  
  def numEpochs = epochData.data.size
  
  override def toString() = {
    s"$pos / $neg"
  }
}