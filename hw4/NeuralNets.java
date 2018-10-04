import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class NeuralNets {
	
	public static class Node {
		Double value = null;
		Double error = null;
		ArrayList<Double> inEdges = null;
		ArrayList<Double> outEdges = null;
		
		public Node() {
			value = 0.0;
			inEdges = new ArrayList<>();
			outEdges = new ArrayList<>();
		}
	}
	
	final static String magicTainPath = "/Users/yashtrivedi/eclipse-workspace/logistic/magic_train.arff";
	final static String magicTestPath = "/Users/yashtrivedi/eclipse-workspace/logistic/magic_test.arff";
	final static String diabetesTrainPath = "/Users/yashtrivedi/eclipse-workspace/logistic/diabetes_train.arff";
	final static String diabetesTestPath = "/Users/yashtrivedi/eclipse-workspace/logistic/diabetes_test.arff";
	
	static ArrayList<Instance> instances = new ArrayList<>();
	static ArrayList<Instance> instancesTest = new ArrayList<>();
	static ArrayList<Attribute> attributes = new ArrayList<>();
	static Attribute classAttribute = null;
	
	static ArrayList<Node> inputLayer = new ArrayList<>();
	static ArrayList<Node> hiddenLayer = new ArrayList<>();
	static Node outputLayer = new Node();
	
	static NumberFormat formatter = new DecimalFormat("#0.000000000");     
	
	static Double sigmoid(Double x) {
		return (Double) 1.0 / (Double)(1.0 + Math.exp(-x));
	}
	
	static ArrayList<Instance> Normalize(ArrayList<Instance> instanceList) {
		for(Attribute a : attributes) {
			if(a.isNominal()) {
				continue;
			} else {
				Double mean = 0.0, sd = 0.0;
				
				for(Instance i : instanceList) {
					mean += i.value(a);
				}
				mean = mean / instanceList.size();
				
				for(Instance i : instanceList) {
					sd += Math.pow((i.value(a) - mean), 2);
				}
				sd = sd / instanceList.size();
				sd = Math.sqrt(sd);
				
				for(Instance i : instanceList) {
					i.setValue(a, (i.value(a) - mean) / sd);
				}
			}
		}
		
		return instanceList;
	}
	
	public static void main(String[] args) throws InterruptedException {
		// TODO Auto-generated method stub
		Integer e = null, h = null;
		Double l = null;
		String inputTrain = args[3], inputTest = args[4];
		
		try {
			l = Double.parseDouble(args[0]);
			h = Integer.parseInt(args[1]);
			e = Integer.parseInt(args[2]);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inputTrain));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			classAttribute = data.classAttribute();
			instances = Collections.list(data.enumerateInstances());
			attributes = Collections.list(data.enumerateAttributes());
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inputTest));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			instancesTest = Collections.list(data.enumerateInstances());
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		instances = Normalize(instances);
		instancesTest = Normalize(instancesTest);
		
		int size = 0;
		for(Attribute a : attributes) {
			if(a.isNominal()) {
				size += a.numValues();
			} else {
				size += 1;
			}
		}
		
		for(int i = 0; i < size; i++) {
			inputLayer.add(new Node());
		}
		
		for(int i = 0; i < h; i++) {
			hiddenLayer.add(new Node());
		}
		
		// Add bias weight for all hidden layer nodes
		for(Node hl : hiddenLayer) {
			Random r = new Random();
			Double x = (-0.01 + (0.01 + 0.01) * (r.nextDouble()));
			hl.inEdges.add(x);
		}
		
		// Add bias weight for output layer node
		{
			Random r = new Random();
			Double x = (-0.01 + (0.01 + 0.01) * (r.nextDouble()));
			outputLayer.inEdges.add(x);
		}
		
		// Add weights for all fully connected edges between input layer and hidden layer.
		for(Node il : inputLayer) {
			for(Node hl : hiddenLayer) {
				Random r = new Random();
				Double x = (-0.01 + (0.01 + 0.01) * (r.nextDouble()));
				il.outEdges.add(x);
				hl.inEdges.add(x);
			}
		}
		
		// Add weights for all fully connected edges between hidden layer and output layer.
		for(Node hl : hiddenLayer) {
			Random r = new Random();
			Double x = (-0.01 + (0.01 + 0.01) * (r.nextDouble()));
			hl.outEdges.add(x);
			outputLayer.inEdges.add(x);
		}
		
		// Run forward propagation and back propagation.
		for(int eCount = 1; eCount <= e; eCount++) {
			int nCorrect = 0, nIncorrect = 0;
			ArrayList<Double> outputs = new ArrayList<>();
			Collections.shuffle(instances);
			
			for(Instance j : instances) {
				
				// Set input layer nodes' input values.
				int count = 0;
				for(Attribute a : attributes) {
					if(a.isNominal()) {
						ArrayList<Object> values = Collections.list(a.enumerateValues());
						for(Object v : values) {
							if(v.toString().equalsIgnoreCase(j.stringValue(a))) {
								inputLayer.get(count++).value = 1.0;
							} else {
								inputLayer.get(count++).value = 0.0;
							}
						}
					} else {
						inputLayer.get(count++).value = j.value(a);
					}
				}
				
				// Compute activations for the hidden layer nodes.
				for(Node hl : hiddenLayer) {
					Double output = hl.inEdges.get(0);
					for(int i = 0; i < inputLayer.size(); i++) {
						output += (inputLayer.get(i).value * hl.inEdges.get(i+1)); 
					}
					output = sigmoid(output);
					hl.value = output;
				}
				
				// Compute activation for the output layer node.
				{
					Double output = outputLayer.inEdges.get(0);
					for(int i = 0; i < hiddenLayer.size(); i++) {
						output += (hiddenLayer.get(i).value * outputLayer.inEdges.get(i+1));
					}
					output = sigmoid(output);
					outputLayer.value = output;
				}
				
				outputs.add(outputLayer.value);
				
				if((outputLayer.value <= 0.5 && j.value(classAttribute) == 0.0) || (outputLayer.value > 0.5 && j.value(classAttribute) == 1.0)){
					nCorrect++;
				} else {
					nIncorrect++;
				}
				
				// Compute error for the output layer node and update bias edge weight.
				{
					outputLayer.error = j.value(classAttribute) - outputLayer.value;
					Double x = outputLayer.inEdges.get(0) + (outputLayer.error * l * 1.0);
					outputLayer.inEdges.set(0, x);
				}
				
				// Compute error for the hidden layer nodes and update weights.
				int hlIndex = 0;
				for(Node hl : hiddenLayer) {
					hl.error = hl.value * (1 - hl.value) * (outputLayer.error * hl.outEdges.get(0));
					
					// Update the bias incoming edge weight.
					{
						Double x = hl.inEdges.get(0) + (hl.error * l * 1.0);
						hl.inEdges.set(0, x);
					}
					
					// Update all the outgoing edge weights.
					for(int i = 0; i < hl.outEdges.size(); i++) {
						Double x = hl.outEdges.get(i) + (l * outputLayer.error * hl.value);
						hl.outEdges.set(i, x);
						outputLayer.inEdges.set(hlIndex+1, x);
					}
					hlIndex++;
				}
				
				// Update all the edges from the input layer to the hidden layer.
				int ilIndex = 0;
				for(Node il : inputLayer) {
					for(int i = 0; i < il.outEdges.size(); i++) {
						Double x = il.outEdges.get(i) + (l * hiddenLayer.get(i).error * il.value);
						il.outEdges.set(i, x);
						hiddenLayer.get(i).inEdges.set(ilIndex + 1, x);
					}
					ilIndex++;
				}
			}
			
			int count = 0;
			Double error = 0.0;
			for(Instance j : instances) {
				Double t = ((-j.value(classAttribute) * Math.log(outputs.get(count))) - ((1 - j.value(classAttribute)) * Math.log(1 - outputs.get(count))));
				error += t;
				count++;
			}
			
			System.out.println(eCount + "\t" + formatter.format(error) + "\t" + nCorrect + "\t" + nIncorrect);
		}
		
		ArrayList<Double> predictions = new ArrayList<>();
		Collections.shuffle(instancesTest);
		for(Instance j : instancesTest) {
			int count = 0;
			for(Attribute a : attributes) {
				if(a.isNominal()) {
					ArrayList<Object> values = Collections.list(a.enumerateValues());
					for(Object v : values) {
						if(v.toString().equalsIgnoreCase(j.stringValue(a))) {
							inputLayer.get(count++).value = 1.0;
						} else {
							inputLayer.get(count++).value = 0.0;
						}
					}
				} else {
					inputLayer.get(count++).value = j.value(a);
				}
			}
			
			// Compute activations for the hidden layer nodes.
			for(Node hl : hiddenLayer) {
				Double output = hl.inEdges.get(0);
				for(int i = 0; i < inputLayer.size(); i++) {
					output += (inputLayer.get(i).value * hl.inEdges.get(i+1)); 
				}
				output = sigmoid(output);
				hl.value = output;
			}
			
			// Compute activation for the output layer node.
			{
				Double output = outputLayer.inEdges.get(0);
				for(int i = 0; i < hiddenLayer.size(); i++) {
					output += (hiddenLayer.get(i).value * outputLayer.inEdges.get(i+1));
				}
				output = sigmoid(output);
				outputLayer.value = output;
			}
			predictions.add(outputLayer.value);
		}
		
		int fp = 0, tp = 0, fn = 0, tn = 0;
		for(int i = 0; i < predictions.size(); i++) {
			int prediction;
			if(predictions.get(i) <= 0.5) {
				prediction = 0;
			} else {
				prediction = 1;
			}
			
			if(((Double)instancesTest.get(i).value(classAttribute)).intValue() == prediction && prediction == 0) {
				tn++;
			} else if(((Double)instancesTest.get(i).value(classAttribute)).intValue() == prediction && prediction == 1) {
				tp++;
			} else if(((Double)instancesTest.get(i).value(classAttribute)).intValue() != prediction && prediction == 0) {
				fn++;
			} else if(((Double)instancesTest.get(i).value(classAttribute)).intValue() != prediction && prediction == 1) {
				fp++;
			}
			
			System.out.println(formatter.format(predictions.get(i)) + "\t" + prediction + "\t" + ((Double)instancesTest.get(i).value(classAttribute)).intValue());
		}
		
		Double precision = (double) tp / (double) (tp + fp);
		Double recall = (double) tp / (double) (tp + fn);
		
		Double f1 = (2 * precision * recall) / (precision + recall);
		System.out.println((tp + tn) + "\t" + (fp + fn));
		System.out.println(formatter.format(f1));
	}
}