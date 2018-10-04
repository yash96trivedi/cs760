import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Bayes {
	
	final static String trainPath = "/Users/yashtrivedi/eclipse-workspace/bayes/lymph_train.arff";
	final static String testPath = "/Users/yashtrivedi/eclipse-workspace/bayes/lymph_test.arff";
	final static String cvPath = "/Users/yashtrivedi/eclipse-workspace/bayes/lymph_cv.arff";
	
	static ArrayList<Instance> instances = new ArrayList<>();
	static ArrayList<Attribute> attributes = new ArrayList<>();
	static ArrayList<Instance> instancesTest = new ArrayList<>();
	static ArrayList<Instance> instancesCV = new ArrayList<>();
	static ArrayList< ArrayList<Instance> > folds = new ArrayList<>();
	
	static double[][] weights = null;
	static int[] parents = null;
	static double[] values = null;
	
	static Attribute classAttribute = null;
	static int countPos = 0, countNeg = 0;
	static int nAttributes = 0;
	
	static Double pYPos = 0.0, pYNeg = 0.0;
	
	static NumberFormat formatter = new DecimalFormat("#0.000000000000");
	
	static String posLabel, negLabel; 
	
	static Double naiveMean = 0.0, tanMean = 0.0;
	
	static void naiveBayes() {
		
		HashMap< String, HashMap<String, Double> > conditionalProbPos = new HashMap<>();
		HashMap< String, HashMap<String, Double> > conditionalProbNeg = new HashMap<>();
		
		for (Attribute a : attributes) {
			
			HashMap<String, Double> pos = new HashMap<>();
			HashMap<String, Double> neg = new HashMap<>();
			
			ArrayList<Object> values = Collections.list(a.enumerateValues());
			
			for(Object v : values) {
				int condCountPos = 0, condCountNeg = 0;
				for (Instance i : instances) {
					if(i.stringValue(a).equalsIgnoreCase(v.toString()) && i.stringValue(i.classAttribute()).equalsIgnoreCase(posLabel)) {
						condCountPos++;
					}
					if(i.stringValue(a).equalsIgnoreCase(v.toString()) && i.stringValue(i.classAttribute()).equalsIgnoreCase(negLabel)) {
						condCountNeg++;
					}
				}
					
				Double posProb = (double)(condCountPos + 1) / (double)(countPos + values.size());
				Double negProb = (double)(condCountNeg + 1) / (double)(countNeg + values.size());
					
				pos.put(v.toString(), posProb);
				neg.put(v.toString(), negProb);
			}
			
			conditionalProbPos.put(a.name(), pos);
			conditionalProbNeg.put(a.name(), neg);
		}
		
		for(Attribute a : attributes) {
			System.out.println(a.name() + " " + classAttribute.name());
		}
		System.out.println();
		
		int nCorrect = 0;
		for(Instance i : instancesTest) {
			double probPos = pYPos, probNeg = pYNeg;
			for(int j = 0; j < i.numAttributes() - 1; j++) {
				
				String attributeName = i.attribute(j).name();
				String attributeValue = i.stringValue(j);
				
				probPos *= conditionalProbPos.get(attributeName).get(attributeValue);
				probNeg *= conditionalProbNeg.get(attributeName).get(attributeValue);
			}
			
			if(probPos >= probNeg) {
				System.out.println(posLabel + " " + i.stringValue(classAttribute) + " " + formatter.format(probPos / (probPos + probNeg)));
				if(posLabel.equalsIgnoreCase(i.stringValue(classAttribute)))
					nCorrect++;
			} else {
				System.out.println(negLabel + " " + i.stringValue(classAttribute) + " " + formatter.format(probNeg / (probPos + probNeg)));
				if(negLabel.equalsIgnoreCase(i.stringValue(classAttribute)))
					nCorrect++;
			}
		}
		
		System.out.println();
		System.out.println(nCorrect);
		
		Double accuracy = (double) nCorrect / (double) instancesTest.size();
//		System.out.println("Naive Accuracy: " + accuracy);
		naiveMean += (accuracy);
	}
	
	static void computeWeights() {
		weights = new double[nAttributes][nAttributes];
		for(int i = 0; i < nAttributes; i++) {
			for(int j = 0; j < nAttributes; j++) {
				if(i == j) {
					weights[i][j] = -1;
				} else {
					double cmi = 0.0;
					
					ArrayList<Object> valuesI = Collections.list(attributes.get(i).enumerateValues());
					for(Object p : valuesI) {
						
						ArrayList<Object> valuesJ = Collections.list(attributes.get(j).enumerateValues());
						for(Object q : valuesJ) {
							
							ArrayList<Object> valuesK = Collections.list(classAttribute.enumerateValues());
							for(Object r : valuesK) {
								
								int c_xi_xj_y = 0, c_xi = 0, c_xj = 0, c_y = 0;
								double p_xi_xj_y = 0.0, p_xi_xj = 0.0, p_xi = 0.0, p_xj = 0.0;
								for(Instance x : instances) {
									if(x.stringValue(attributes.get(i)).equalsIgnoreCase(p.toString()) 
										&& x.stringValue(attributes.get(j)).equalsIgnoreCase(q.toString())
										&& x.stringValue(classAttribute).equalsIgnoreCase(r.toString()))
										c_xi_xj_y++;
									
									if(x.stringValue(attributes.get(i)).equalsIgnoreCase(p.toString())
										&& x.stringValue(classAttribute).equalsIgnoreCase(r.toString()))
										c_xi++;
									
									if(x.stringValue(attributes.get(j)).equalsIgnoreCase(q.toString())
										&& x.stringValue(classAttribute).equalsIgnoreCase(r.toString()))
										c_xj++;
									
									if(x.stringValue(classAttribute).equalsIgnoreCase(r.toString()))
										c_y++;
								}
								
								p_xi_xj_y = (double) (c_xi_xj_y + 1) / (double) (instances.size() + (valuesI.size() * valuesJ.size() * valuesK.size()));
								p_xi_xj = (double) (c_xi_xj_y + 1) / (double) (c_y + valuesI.size() * valuesJ.size());
								p_xi = (double) (c_xi + 1) / (double) (c_y + valuesI.size());
								p_xj = (double) (c_xj + 1) / (double) (c_y + valuesJ.size());
								
								cmi += (p_xi_xj_y * (Math.log((p_xi_xj) / ((p_xi) * (p_xj))) / Math.log(2.0)));
							}
						}
					}
					weights[i][j] = cmi;
				}
			}
		}
	}
	
	static void primMST() {
		parents = new int[nAttributes];
		values = new double[nAttributes];
		boolean[] mst = new boolean[nAttributes];
		
		for(int i = 0; i < nAttributes; i++) {
			values[i] = Double.NEGATIVE_INFINITY;
			mst[i] = false;
		}

		values[0] = 0;
		parents[0] = -1;
		
		for(int i = 0; i < nAttributes - 1; i++) {
			int idx = -1;
			double value = Double.NEGATIVE_INFINITY;
			
			for(int j = 0; j < nAttributes; j++) {
				if(mst[j] == false && values[j] > value) {
					value = values[j];
					idx = j;
				}
			}
			
			mst[idx] = true;
			
			for(int j = 0; j < nAttributes; j++) {
				if(mst[j] == false && weights[idx][j] > values[j]) {
					parents[j] = idx;
					values[j] = weights[idx][j];
				}
			}
		}
	}
	
	static void taNaiveBayes() {
		
		HashMap< String, HashMap<String, Double> > conditionalProbPos = new HashMap<>();
		HashMap< String, HashMap<String, Double> > conditionalProbNeg = new HashMap<>();
		
		for (Attribute a : attributes) {
			
			HashMap<String, Double> pos = new HashMap<>();
			HashMap<String, Double> neg = new HashMap<>();
			
			Attribute parent = null;
			ArrayList<Object> parentValues = null;
			
			if(parents[a.index()] != -1) {
				parent = attributes.get(parents[a.index()]);
			}
			
			ArrayList<Object> values = Collections.list(a.enumerateValues());
			if(parent != null) {
				parentValues = Collections.list(parent.enumerateValues());
			}
			
			for(Object v : values) {
				if(parent != null) {
					for(Object pValue : parentValues) {
						int condCountPos = 0, condCountNeg = 0, posTotalCount = 0, negTotalCount = 0;
						for (Instance i : instances) {
							if(i.stringValue(a).equalsIgnoreCase(v.toString())
									&& i.stringValue(parent).equalsIgnoreCase(pValue.toString())
									&& i.stringValue(i.classAttribute()).equalsIgnoreCase(posLabel)) {
								condCountPos++;
							}
						
							if(i.stringValue(a).equalsIgnoreCase(v.toString()) 
									&& i.stringValue(parent).equalsIgnoreCase(pValue.toString())
									&& i.stringValue(i.classAttribute()).equalsIgnoreCase(negLabel)) {
								condCountNeg++;
							}
							
							if(i.stringValue(parent).equalsIgnoreCase(pValue.toString())
									&& i.stringValue(i.classAttribute()).equalsIgnoreCase(posLabel)) {
								posTotalCount++;
							}
							
							if(i.stringValue(parent).equalsIgnoreCase(pValue.toString())
									&& i.stringValue(i.classAttribute()).equalsIgnoreCase(negLabel)) {
								negTotalCount++;
							}
						}
					
						Double posProb = (double)(condCountPos + 1) / (double)(posTotalCount + values.size());
						Double negProb = (double)(condCountNeg + 1) / (double)(negTotalCount + values.size());
					
						pos.put(v.toString() + pValue.toString(), posProb);
						neg.put(v.toString() + pValue.toString(), negProb);
					}
					
					conditionalProbPos.put(a.name() + parent.name(), pos);
					conditionalProbNeg.put(a.name() + parent.name(), neg);
					
				} else {
					int condCountPos = 0, condCountNeg = 0;
					for (Instance i : instances) {
						if(i.stringValue(a).equalsIgnoreCase(v.toString()) && i.stringValue(i.classAttribute()).equalsIgnoreCase(posLabel)) {
							condCountPos++;
						}
						if(i.stringValue(a).equalsIgnoreCase(v.toString()) && i.stringValue(i.classAttribute()).equalsIgnoreCase(negLabel)) {
							condCountNeg++;
						}
					}
							
					Double posProb = (double)(condCountPos + 1) / (double)(countPos + values.size());
					Double negProb = (double)(condCountNeg + 1) / (double)(countNeg + values.size());
							
					pos.put(v.toString(), posProb);
					neg.put(v.toString(), negProb);
					
					conditionalProbPos.put(a.name(), pos);
					conditionalProbNeg.put(a.name(), neg);
				}
			}
		}
		
		for(Attribute a : attributes) {
			if(parents[a.index()] == -1) {
				System.out.println(a.name() + " " + classAttribute.name());
			} else {
				System.out.println(a.name() + " " + attributes.get(parents[a.index()]).name() + " " + classAttribute.name());
			}
		}
		System.out.println();
		
		int nCorrect = 0;
		for(Instance i : instancesTest) {
			double probPos = pYPos, probNeg = pYNeg;
			for(int j = 0; j < i.numAttributes() - 1; j++) {
				
				Attribute a = i.attribute(j);
				String attributeName = a.name();
				String attributeValue = i.stringValue(j);
				
				Attribute parent = null;
				String parentName = "";
				String parentValue = "";
				
				if(parents[a.index()] != -1) {
					parent = attributes.get(parents[a.index()]);
					parentName = parent.name();
					parentValue = i.stringValue(parent.index());
				}
				
				probPos *= conditionalProbPos.get(attributeName + parentName).get(attributeValue + parentValue);
				probNeg *= conditionalProbNeg.get(attributeName + parentName).get(attributeValue + parentValue);
			}
			
			if(probPos >= probNeg) {
				System.out.println(posLabel + " " + i.stringValue(classAttribute) + " " + formatter.format(probPos / (probPos + probNeg)));
				if(posLabel.equalsIgnoreCase(i.stringValue(classAttribute)))
					nCorrect++;
			} else {
				System.out.println(negLabel + " " + i.stringValue(classAttribute) + " " + formatter.format(probNeg / (probPos + probNeg)));
				if(negLabel.equalsIgnoreCase(i.stringValue(classAttribute)))
					nCorrect++;
			}
		}
		
		System.out.println();
		System.out.println(nCorrect);
		
		Double accuracy = (double) nCorrect / (double) instancesTest.size();
//		System.out.println("TAN Accuracy: " + accuracy);
		tanMean += (accuracy);
	}
	
//	static void getPlotValues() {
//		
//		ArrayList<String> trueValue = new ArrayList<>();
//		ArrayList<String> predictedValue = new ArrayList<>();
//		ArrayList<Double> probability = new ArrayList<>();
//		
//		String inputFileOneSorted = "/Users/yashtrivedi/Documents/UW-Madison/cs760/hw-bayes/naive_output_sorted.txt";
//		String inputFileTwoSorted = "/Users/yashtrivedi/Documents/UW-Madison/cs760/hw-bayes/tan_output_sorted.txt";
//		
//		String inputFileOne = "";
//		String inputFileTwo = "";
//		
//		try {
//			Scanner sc = new Scanner(new File(inputFileTwoSorted));
//			while (sc.hasNextLine()) {
//				String line = sc.nextLine();
//				trueValue.add(line.split("\\s+")[0]);
//				predictedValue.add(line.split("\\s+")[1]);
//				probability.add(Double.parseDouble(line.split("\\s+")[2]));
//			}	
//		} catch (Exception ex) {
//			ex.printStackTrace();
//		}
//		
//		for(Double x : probability) {
//			int tp = 0, fp = 0, tn = 0, fn = 0;
//			for(int i = 0; i < predictedValue.size(); i++) {
//				if(Double.compare(probability.get(i), x) >= 0 && predictedValue.get(i).compareToIgnoreCase(posLabel) == 0) {
//					tp++;
//				} else if(Double.compare(probability.get(i), x) < 0 && predictedValue.get(i).compareToIgnoreCase(negLabel) == 0) {
//					tn++;
//				} else if(Double.compare(probability.get(i), x) < 0 && predictedValue.get(i).compareToIgnoreCase(posLabel) == 0) {
//					fn++;
//				} else if(Double.compare(probability.get(i), x) >= 0 && predictedValue.get(i).compareToIgnoreCase(negLabel) == 0) {
//					fp++;
//				}
//			}
//			
//			Double precision = (double) tp / (double) (tp + fp);
//			Double recall = (double) tp / (double) (tp + fn);
//			
////			System.out.println("TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
////			System.out.print(recall + ", ");
//		}
//	}
	
	static void generateFolds() {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(cvPath));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			instancesCV = Collections.list(data.enumerateInstances());
//			int initialSize = instancesCV.size();

			for(int i = 0; i < 2; i++) {
				ArrayList<Instance> temp = new ArrayList<>();
				for(int j = 0; j < 15; j++) {
					Random rand = new Random();
					int n = rand.nextInt(instancesCV.size());
					temp.add(instancesCV.get(n));
					instancesCV.remove(n);
				}
				folds.add(temp);
			}
			
			for(int i = 0; i < 8; i++) {
				ArrayList<Instance> temp = new ArrayList<>();
				for(int j = 0; j < 14; j++) {
					Random rand = new Random();
					int n = rand.nextInt(instancesCV.size());
					temp.add(instancesCV.get(n));
					instancesCV.remove(n);
				}
				folds.add(temp);
			}
			
//			int sum = 0;
//			for(int i = 0; i < folds.size(); i++) {
//				System.out.println(folds.get(i).size());
//				sum += folds.get(i).size();
//			}
//			
//			System.out.println(sum + " " + initialSize);
			
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	public static void main(String args[]) {
		
		String inputTrain = args[0], inputTest = args[1], c = args[2];
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inputTrain));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			classAttribute = data.classAttribute();
			instances = Collections.list(data.enumerateInstances());
			attributes = Collections.list(data.enumerateAttributes());
			nAttributes = attributes.size();
			
			posLabel = classAttribute.value(0);
			negLabel = classAttribute.value(1);
			
			for (Instance i : instances) {
				if(i.stringValue(i.classAttribute()).equalsIgnoreCase(posLabel))
					countPos++;
				if(i.stringValue(i.classAttribute()).equalsIgnoreCase(negLabel))
					countNeg++;
			}
			
			reader = new BufferedReader(new FileReader(inputTest));
			arff = new ArffReader(reader);
			data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);

			instancesTest = Collections.list(data.enumerateInstances());
			
			pYPos = (double) (countPos + 1) / (double) (countPos + countNeg + 2);
			pYNeg = (double) (countNeg + 1) / (double) (countPos + countNeg + 2);
			
//			getPlotValues();
//			generateFolds();
			if(c.compareToIgnoreCase("n") == 0) {
				naiveBayes();
			} else if(c.compareToIgnoreCase("t") == 0) {
				computeWeights();
				primMST();
				taNaiveBayes();
			}
			
//			for(int i = 0; i < 10; i++) {
//				instances.clear();
//				instancesTest.clear();
//				
//				for(int j = 0; j < 10; j++) {
//					if(i == j) {
//						instancesTest.addAll(folds.get(j));
//					} else {
//						instances.addAll(folds.get(j));
//					}
//				}
//				
//				for (Instance x : instances) {
//					if(x.stringValue(x.classAttribute()).equalsIgnoreCase(posLabel))
//						countPos++;
//					if(x.stringValue(x.classAttribute()).equalsIgnoreCase(negLabel))
//						countNeg++;
//				}
//				
//				pYPos = (double) (countPos + 1) / (double) (countPos + countNeg + 2);
//				pYNeg = (double) (countNeg + 1) / (double) (countPos + countNeg + 2);
//				
//				naiveBayes();
//				
//				computeWeights();
//				primMST();
//				taNaiveBayes();
//			}
//			
//			System.out.println();
//			System.out.println("Naive Mean: " + naiveMean / (double) 10);
//			System.out.println("TAN Mean: " + tanMean / (double) 10);
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
