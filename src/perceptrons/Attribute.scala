package perceptrons

class Attribute (var name: String, var weight: Double, var input: Double = 1.0){
  
  def calc() = weight * input 
  
  override def toString() = s"$name with weight: $weight and input $input"
}