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

public class Logistic {

	final static String magicTainPath = "/Users/yashtrivedi/eclipse-workspace/logistic/magic_train.arff";
	final static String magicTestPath = "/Users/yashtrivedi/eclipse-workspace/logistic/magic_test.arff";
	final static String diabetesTrainPath = "/Users/yashtrivedi/eclipse-workspace/logistic/diabetes_train.arff";
	final static String diabetesTestPath = "/Users/yashtrivedi/eclipse-workspace/logistic/diabetes_test.arff";
	
	static ArrayList<Instance> instances = new ArrayList<>();
	static ArrayList<Instance> instancesTest = new ArrayList<>();
	static ArrayList<Attribute> attributes = new ArrayList<>();
	static Attribute classAttribute = null;
	
	static ArrayList<Double> inputs = null;
	static ArrayList<Double> outputs = null;
	static ArrayList<Double> weights = null;
	
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
	
	public static void main(String[] args) {
		
		Integer e = null;
		Double l = null;
		String inputTrain = args[2], inputTest = args[3];
		
		try {
			l = Double.parseDouble(args[0]);
			e = Integer.parseInt(args[1]);
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
		
		weights = new ArrayList<>(size);
		for(int i = 0; i <= size; i++) {
			Random r = new Random();
			Double x = (-0.01 + (0.01 + 0.01) * (r.nextDouble()));
			weights.add(x);
		}
		
		for(int i = 1; i <= e; i++) {
			int nCorrect = 0, nIncorrect = 0;
			outputs = new ArrayList<>();
			Collections.shuffle(instances);
			for(Instance j : instances) {
				inputs = new ArrayList<>(size);
				inputs.add(1.0);
				for(Attribute a : attributes) {
					if(a.isNominal()) {
						ArrayList<Object> values = Collections.list(a.enumerateValues());
						for(Object v : values) {
							if(v.toString().equalsIgnoreCase(j.stringValue(a))) {
								inputs.add(1.0);
							} else {
								inputs.add(0.0);
							}
						}
					} else {
						inputs.add(j.value(a));
					}
				}
				
				Double output = 0.0;
				for(int k = 0; k < inputs.size(); k++) {
					output += (inputs.get(k) * weights.get(k));
				}
				
				Double outputSigmoid = sigmoid(output);
				outputs.add(outputSigmoid);
				
				if((outputSigmoid <= 0.5 && j.value(classAttribute) == 0.0) || (outputSigmoid > 0.5 && j.value(classAttribute) == 1.0)){
					nCorrect++;
				} else {
					nIncorrect++;
				}
				
				Double dError = (outputSigmoid - j.value(classAttribute));
				for(int k = 0; k < weights.size(); k++) {
					 Double deltaW = (-l * (dError) * inputs.get(k));
					 Double x = weights.get(k) + deltaW;
					 weights.set(k, x);
				}
			}
			
			int count = 0;
			Double error = 0.0;
			for(Instance j : instances) {
				Double t = ((-j.value(classAttribute) * Math.log(outputs.get(count))) - ((1 - j.value(classAttribute)) * Math.log(1 - outputs.get(count))));
				error += t;
				count++;
			}
			
			System.out.println(i + "\t" + formatter.format(error) + "\t" + nCorrect + "\t" + nIncorrect);
		}
		
		ArrayList<Double> predictions = new ArrayList<>();
		Collections.shuffle(instancesTest);
		for(Instance j : instancesTest) {
			inputs = new ArrayList<>(size);
			inputs.add(1.0);
			for(Attribute a : attributes) {
				if(a.isNominal()) {
					ArrayList<Object> values = Collections.list(a.enumerateValues());
					for(Object v : values) {
						if(v.toString().equalsIgnoreCase(j.stringValue(a))) {
							inputs.add(1.0);
						} else {
							inputs.add(0.0);
						}
					}
				} else {
					inputs.add(j.value(a));
				}
			}
			
			Double output = 0.0;
			for(int k = 0; k < inputs.size(); k++) {
				output += (inputs.get(k) * weights.get(k));
			}
			
			Double outputSigmoid = sigmoid(output);
			predictions.add(outputSigmoid);
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
