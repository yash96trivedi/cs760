import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;

import weka.core.Attribute;
import weka.core.Instance;

public class Node {
	
	int nPos, nNeg;
	ArrayList<Attribute> attributes = new ArrayList<>();
	ArrayList<Instance> examples = new ArrayList<>();
	Double entropy; 
	Attribute decisionAttribute = null;
	ArrayList<Node> children = new ArrayList<>();
	Boolean isLeaf = false;
	String label = null;
	String parentLabel = null;
	Double splitValue = null;
	
	Attribute myAttribute = null;
	String myVal= null;
	Double mySplitVal = null;
	String myPrefix = null;
	
	NumberFormat formatter = new DecimalFormat("#0.000000");
	
	public Node(ArrayList<Instance> instances, String pLabel, Attribute dAttribute, String value, Double splitVal, String pre) {
		
		attributes.addAll(DTree.attributeList);
		examples.addAll(instances);
		parentLabel = pLabel;
		
		myAttribute = dAttribute;
		myVal = value;
		mySplitVal = splitVal;
		myPrefix = pre;
		
		for(Instance x : examples) {
			if(x.toString(x.classAttribute()).equalsIgnoreCase("+")) {
				nPos++;
			} else {
				nNeg++;
			}
		}
		entropy = DTree.calcEntropy(nPos, nNeg);
	}
	
	public void getBestAttribute() {

		if(nPos > nNeg) {
			label = "+";
		} else if(nNeg > nPos) {
			label = "-";
		} else {
			label = parentLabel;
		}
		
		boolean flagPos = false, flagNeg = false;
		if(examples.size() < DTree.m) {
			isLeaf = true;
			
			if(myAttribute.equals(DTree.cAttribute)) {
				return ;
			} else {
				if(myAttribute.isNumeric())
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " " + myVal + " " + formatter.format(mySplitVal) + " [" + nPos + " " + nNeg + "]: " + label);
				else
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " = " + myVal + " [" + nPos + " " + nNeg + "]: " + label);
			}
			return ;
		}
		
		for(Instance x :  examples) {
			if(x.toString(x.classAttribute()).equalsIgnoreCase("+")) {
				flagPos = true;
			} else if(x.toString(x.classAttribute()).equalsIgnoreCase("-")) {
				flagNeg = true;
			}
		}
		
		if(!flagNeg || !flagPos) {
			isLeaf = true;
			if(myAttribute.equals(DTree.cAttribute)) {
				return ;
			} else {
				if(myAttribute.isNumeric())
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " " + myVal + " " + formatter.format(mySplitVal) + " [" + nPos + " " + nNeg + "]: " + label);
				else
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " = " + myVal + " [" + nPos + " " + nNeg + "]: " + label);
			}
			return ;
		}
		
		double infoGain = 0.0, maxInfoGain = 0.0;
		Attribute bestAttribute = null;
		ArrayList<Object> values = new ArrayList<>();
		
		for(Attribute a : attributes) {
			double newEntropy = 0.0;
			
			if(a.isNumeric()) {
				
				ArrayList<Double> splitThresh = new ArrayList<>();
				ArrayList<Double> splitThreshMid = new ArrayList<>();
				
				for(Instance x : examples) {
					splitThresh.add(x.value(a));
				}
				Collections.sort(splitThresh);
				for(int i = 0; i < splitThresh.size() - 1; i++) {
					splitThreshMid.add((splitThresh.get(i) + splitThresh.get(i+1)) / 2);
				}
				
				for(int i = 0; i < splitThreshMid.size(); i++) {
					int pos1 = 0, neg1 = 0, pos2 = 0, neg2 = 0;
					newEntropy = 0.0;
					for(Instance x : examples) {
						if((Double.compare(x.value(a), splitThreshMid.get(i)) <= 0) && (x.toString(x.classAttribute()).equalsIgnoreCase("+"))) {
							pos1++;
						} else if((Double.compare(x.value(a), splitThreshMid.get(i)) <= 0) && (x.toString(x.classAttribute()).equalsIgnoreCase("-"))){
							neg1++;
						} else if((Double.compare(x.value(a), splitThreshMid.get(i)) > 0) && (x.toString(x.classAttribute()).equalsIgnoreCase("+"))) {
							pos2++;
						} else if((Double.compare(x.value(a), splitThreshMid.get(i)) > 0) && (x.toString(x.classAttribute()).equalsIgnoreCase("-"))){
							neg2++;
						}
					}
					
					newEntropy = (((double)(pos1+neg1)/(double)(examples.size())) * DTree.calcEntropy(pos1, neg1))
							+ (((double)(pos2+neg2)/(double)(examples.size())) * DTree.calcEntropy(pos2, neg2));
					
					infoGain = entropy - newEntropy;
					if(infoGain > maxInfoGain) {
						bestAttribute = a;
						splitValue = splitThreshMid.get(i);
						maxInfoGain = infoGain; 
					}
				}
				
			} else {
				
				values = Collections.list(a.enumerateValues());
				for(Object v : values) {
					int pos = 0, neg = 0;
					for(Instance x : examples) {
						if((x.toString(a).equalsIgnoreCase(v.toString())) && (x.toString(x.classAttribute()).equalsIgnoreCase("+"))) {
							pos++;
						} else if((x.toString(a).equalsIgnoreCase(v.toString())) && (x.toString(x.classAttribute()).equalsIgnoreCase("-"))){
							neg++;
						}
					}
					newEntropy += (((double)(pos+neg)/(double)(examples.size())) * DTree.calcEntropy(pos, neg));
				}
			}
	
			infoGain = entropy - newEntropy;
			if(infoGain > maxInfoGain) {
				bestAttribute = a;
				maxInfoGain = infoGain;
			}
		}
		
		decisionAttribute = bestAttribute;
		if(decisionAttribute == null) {
			isLeaf = true;
			
			if(myAttribute.equals(DTree.cAttribute)) {
				return ;
			} else {
				if(myAttribute.isNumeric())
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " " + myVal + " " + formatter.format(mySplitVal) + " [" + nPos + " " + nNeg + "]: " + label);
				else
					System.out.println(myPrefix + myAttribute.name().toLowerCase() + " = " + myVal + " [" + nPos + " " + nNeg + "]: " + label);
			}
			return ; 
		}
		
		if(decisionAttribute.isNumeric()) {
			ArrayList<Instance> instances1 = new ArrayList<>();
			ArrayList<Instance> instances2 = new ArrayList<>();
			for(Instance x : examples) {
				if((Double.compare(x.value(decisionAttribute), splitValue) <= 0)) {
					instances1.add(x);
				} else if((Double.compare(x.value(decisionAttribute), splitValue) > 0)) {
					instances2.add(x);
				}
			}
			
			if(myAttribute.equals(DTree.cAttribute)) {
				Node x = new Node(instances1, label, decisionAttribute, "<=", splitValue, myPrefix + "");
				this.children.add(x);
				x = new Node(instances2, label, decisionAttribute, ">", splitValue, myPrefix + "");
				this.children.add(x);
			} else {
				Node x = new Node(instances1, label, decisionAttribute, "<=", splitValue, myPrefix + "|\t");
				this.children.add(x);
				x = new Node(instances2, label, decisionAttribute, ">", splitValue, myPrefix + "|\t");
				this.children.add(x);
			}
			
		} else {
			
			values = Collections.list(decisionAttribute.enumerateValues());
			for(Object v : values) {
				ArrayList<Instance> instances = new ArrayList<>();
				for(Instance x : examples) {
					if(x.toString(decisionAttribute).equalsIgnoreCase(v.toString())) {
						instances.add(x);
					}
				}
				if(myAttribute.equals(DTree.cAttribute)) {
					Node x = new Node(instances, label, decisionAttribute, v.toString(), 0.0, myPrefix + "");
					this.children.add(x);
				} else {
					Node x = new Node(instances, label, decisionAttribute, v.toString(), 0.0, myPrefix + "|\t");
					this.children.add(x);
				}
			}
		}
		
		if(myAttribute.equals(DTree.cAttribute)) {
			return ;
		} else {
			if(myAttribute.isNumeric())
				System.out.println(myPrefix + myAttribute.name().toLowerCase() + " " + myVal + " " + formatter.format(mySplitVal) + " [" + nPos + " " + nNeg + "]");
			else
				System.out.println(myPrefix + myAttribute.name().toLowerCase() + " = " + myVal + " [" + nPos + " " + nNeg + "]");
		}
	}
}
