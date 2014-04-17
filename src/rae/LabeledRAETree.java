package rae;

import org.jblas.*;

import classify.LabeledDatum;

import util.*;

import java.util.*;

public class LabeledRAETree implements LabeledDatum<Double, Integer> {
    RAENode[] T;
    double[] feature;
    Structure structure;
    int sentenceLength, treeSize, Label;
    double totalScore;

    public LabeledRAETree(int SentenceLength, int Label) {
        this.sentenceLength = SentenceLength;
        treeSize = 2 * SentenceLength - 1;
        T = new RAENode[treeSize];
        structure = new Structure(treeSize);
        this.Label = Label;
    }

    public LabeledRAETree(int sentenceLength, int Label, int hiddenSize, DoubleMatrix wordsEmbedded) {
        this(sentenceLength, Label);
        for (int i = 0; i < treeSize; i++) {
            T[i] = new RAENode(i, sentenceLength, hiddenSize, wordsEmbedded);
            structure.add(new Pair<Integer, Integer>(-1, -1));
        }
    }

    public RAENode[] getNodes() {
        return T;
    }

    public LabeledRAETree(int SentenceLength, int Label, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded) {
        this(SentenceLength, Label);
        for (int i = 0; i < treeSize; i++) {
            T[i] = new RAENode(i, SentenceLength, HiddenSize, CatSize, WordsEmbedded);
            structure.add(new Pair<Integer, Integer>(-1, -1));
        }
    }

    public int[] getStructureString() {
        int[] parents = new int[treeSize];
        Arrays.fill(parents, -1);

        for (int i = treeSize - 1; i >= 0; i--) {
            int leftChild = structure.get(i).getFirst();
            int rightChild = structure.get(i).getSecond();
            if (leftChild != -1 && rightChild != -1) {
                if (parents[leftChild] != -1
                        || parents[rightChild] != -1)
                    System.err.println("TreeStructure is messed up!");
                parents[leftChild] = i;
                parents[rightChild] = i;
            }
        }
        return parents;
    }

    @Override
    public String toString() {
        return null;
    }

    @Override
    public Integer getLabel() {
        return Label;
    }


    public double[] getFeaturesVector() {
        if (feature != null)
            return feature;

        int HiddenSize = T[0].features.rows;
        feature = new double[HiddenSize * 2];
        DoubleMatrix tf = new DoubleMatrix(HiddenSize, treeSize);
        if (sentenceLength > 1) {
            for (int i = 0; i < treeSize; i++)
                tf.putColumn(i, T[i].features);
            tf.muli(1.0 / treeSize);

            System.arraycopy(T[2 * sentenceLength - 2].features.data, 0, feature, 0, HiddenSize);
            System.arraycopy(tf.rowSums().data, 0, feature, HiddenSize, HiddenSize);
        } else {
            System.arraycopy(T[2 * sentenceLength - 2].features.data, 0, feature, 0, HiddenSize);
            System.arraycopy(T[2 * sentenceLength - 2].features.data, 0, feature, HiddenSize, HiddenSize);
        }
        return feature;
    }

    @Deprecated
    public Collection<Double> getFeatures() {
        System.err.println("There's no way I am returning a Collection."
                + "\nPlease use the getFeatureVector method instead.");

        return null;
    }
}

class Structure extends ArrayList<Pair<Integer, Integer>> {
    private static final long serialVersionUID = -1616780629111786862L;

    public Structure(int Capacity) {
        super(Capacity);
    }

    public String toString() {
        String retString = "";
        for (Pair<Integer, Integer> pii : this)
            retString += "<" + pii.getFirst() + "," + pii.getSecond() + ">";
        return retString;
    }
}