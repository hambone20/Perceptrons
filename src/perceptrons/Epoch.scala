package perceptrons

import scala.collection.mutable.ArrayBuffer

class Epoch (var data: ArrayBuffer[(Int, Int)] = ArrayBuffer()) {

  def addEpoch(correct: Int, num: Int) = {
    data += (correct -> num)
  }
  
  def getAccuracy(): Double = {
    val (correct: ArrayBuffer[Int], num: ArrayBuffer[Int]) = data.unzip

    return (correct.sum.toDouble / num.sum)
  }
  
  def getLastEpochAccuracy() : Double = {
    val x = data.last
      
    x._1.toDouble / x._2
  }
  
  def epochsDone(): Boolean = { // determine whether to proceed with another epoch
    if (BinaryPerceptrons.doEpochs > 0){ // shortcut
      if (data.size >= BinaryPerceptrons.doEpochs)
        return true
      else
        return false
    }
    if ((data.size == 1 && getLastEpochAccuracy() == 1.0) || data.size >= BinaryPerceptrons.maxEpochs)
      return true
      
    if (data.size < 2)
      return false
    
    val test = data.takeRight(2) // get last two
    
    val epochDiff = test(1)._1.toDouble / test(1)._2 - test(0)._1.toDouble / test(0)._2
    
    if (Math.abs(epochDiff) < BinaryPerceptrons.tolerance)
    	return true
   
    return false
  }
	
  override def toString() = {
    s"$data"
  }
}