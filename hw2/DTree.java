import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class DTree {
	
	static ArrayList<Attribute> attributeList = new ArrayList<>();
	static ArrayList<Instance> instanceList = new ArrayList<>();
	static int m;
	static Attribute cAttribute = null;
	
	public static double calcEntropy(int nPos, int nNeg) {
		
		double entropy = 0.0;
		if(nPos == 0 || nNeg == 0)
			return entropy;
		
		double p1 = ((double)nPos / (double)(nPos + nNeg));
		double l1 = Math.log(p1) / Math.log(2);
		double p2 = ((double)nNeg / (double)(nPos + nNeg));
		double l2 = Math.log(p2) / Math.log(2);
		
		entropy = -p1*l1 -p2*l2;
		
		return entropy;
	}
	
	public static void main(String[] args) {
		
		String inputTrain = args[0], inputTest = args[1];
		m = Integer.parseInt(args[2]);
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inputTrain));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			attributeList = Collections.list(data.enumerateAttributes());
			instanceList = Collections.list(data.enumerateInstances());
			cAttribute = data.classAttribute();
			
//			System.out.println("TOTAL EXAMPLES: " + instanceList.size());
			
			int nPos = 0, nNeg = 0; 
			for(Instance x : instanceList) {
				if(x.toString(x.classAttribute()).equalsIgnoreCase("+")) {
					nPos++;
				} else {
					nNeg++;
				}
			}
			
			String label = null;
			if(nPos > nNeg)
				label = "+";
			else if(nNeg > nPos)
				label = "-";
			
			Node root = new Node(instanceList, label, cAttribute, "###", Double.MIN_VALUE, "");
			Deque<Node> stack = new ArrayDeque<>();
			stack.push(root);
			
			while(!stack.isEmpty()) {
				Node n = stack.pop();
				n.getBestAttribute();
				Collections.reverse(n.children);
				for(Node child : n.children) {
					stack.push(child);
				}
				Collections.reverse(n.children);
			}
			
//			System.out.println(root.decisionAttribute.name());
//			System.out.println(root.splitValue);
			
			reader = new BufferedReader(new FileReader(inputTest));
			arff = new ArffReader(reader);
			data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			instanceList = Collections.list(data.enumerateInstances());
			
			System.out.println("<Predictions for the Test Set Instances>");
			int i = 1;
			int tp = 0, tn = 0, fp = 0, fn = 0;
			int nPosTest = 0, nNegTest = 0;
			
			FileWriter fileWriter = new FileWriter("roc.txt");
		    PrintWriter printWriter = new PrintWriter(fileWriter);
		    
			for(Instance x : instanceList) {
				Node n = root;
				while(!n.isLeaf) {
//					System.out.println(n.decisionAttribute);
					if(n.decisionAttribute.isNumeric()) {
						Double val = x.value(n.decisionAttribute);
						if(Double.compare(val, n.splitValue) <= 0) {
							n = n.children.get(0);
						} else {
							n = n.children.get(1);
						}
					} else {
						String val = x.toString(n.decisionAttribute);
						for(Node a : n.children) {
							if(a.myVal.equalsIgnoreCase(val)) {
								n = a;
							}
						}
					}
				}
				
				if(x.toString(x.classAttribute()).equalsIgnoreCase(n.label) && n.label.equalsIgnoreCase("+")) {
					tp++;
				} else if(x.toString(x.classAttribute()).equalsIgnoreCase(n.label) && n.label.equalsIgnoreCase("-")) {
					tn++;
				} else if(!x.toString(x.classAttribute()).equalsIgnoreCase(n.label) && n.label.equalsIgnoreCase("+")) {
					fp++;
				} else if(!x.toString(x.classAttribute()).equalsIgnoreCase(n.label) && n.label.equalsIgnoreCase("-")) {
					fn++;
				}
				
				if(x.toString(x.classAttribute()).equalsIgnoreCase("+")) {
					nPosTest++;
				} else if(x.toString(x.classAttribute()).equalsIgnoreCase("-")) {
					nNegTest++;
				}
				
				System.out.println(i + ": Actual: " + x.toString(x.classAttribute()) + " Predicted: " + n.label);
				Double confidence = (double)(n.nPos + 1) / ((double)n.nNeg + n.nPos + 2);
				printWriter.println(confidence + " " + x.toString(x.classAttribute()));
				i++;
			}
			printWriter.close();
			System.out.println("Number of correctly classified: " + (tp + tn) + " Total number of test instances: " + (tp + tn + fp + fn));
//			System.out.println("Positive = " + nPosTest + " Negative = " + nNegTest);
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
	}
}
