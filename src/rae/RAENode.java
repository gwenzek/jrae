package rae;

import org.jblas.DoubleMatrix;

import java.util.LinkedList;

public class RAENode {
    RAENode parent, LeftChild, RightChild;
    int NodeName, subtreeSize;
    double[] scores; //, Freq;
    DoubleMatrix unNormalizedFeatures,
            features, LeafFeatures, Z,
            DeltaOut1, DeltaOut2, ParentDelta,
            catDelta, dW1, dW2, dW3, dW4, dL, Y1C1, Y2C2;

    /**
     * Specialized Constructor for fitting in that list
     *
     * @param nodeIndex
     * @param sentenceLength
     * @param HiddenSize
     * @param wordsEmbedded
     */
    public RAENode(int nodeIndex, int sentenceLength, int HiddenSize, DoubleMatrix wordsEmbedded) {
        NodeName = nodeIndex;
        parent = LeftChild = RightChild = null;
        scores = null;
//			Freq = 0;
        subtreeSize = 0;
        if (nodeIndex < sentenceLength) {
            features = wordsEmbedded.getColumn(nodeIndex);
            unNormalizedFeatures = wordsEmbedded.getColumn(nodeIndex);
        }
    }

    public double[] getScores() {
        return scores;
    }

    public double[] getFeatures() {
        return features.data;
    }

    public LinkedList<Integer> getSubtreeWordIndices() {
        LinkedList<Integer> list = new LinkedList<Integer>();
        if (isLeaf())
            list.add(NodeName);
        else {
            list.addAll(LeftChild.getSubtreeWordIndices());
            list.addAll(RightChild.getSubtreeWordIndices());
        }
        return list;
    }

    public RAENode(int NodeIndex, int SentenceLength, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded) {
        this(NodeIndex, SentenceLength, HiddenSize, WordsEmbedded);
        DeltaOut1 = DoubleMatrix.zeros(HiddenSize, 1);
        DeltaOut2 = DoubleMatrix.zeros(HiddenSize, 1);
        ParentDelta = DoubleMatrix.zeros(HiddenSize, 1);
        Y1C1 = DoubleMatrix.zeros(HiddenSize, 1);
        Y2C2 = DoubleMatrix.zeros(HiddenSize, 1);
        if (NodeIndex >= SentenceLength) {
            features = DoubleMatrix.zeros(HiddenSize, 1);
            unNormalizedFeatures = DoubleMatrix.zeros(HiddenSize, 1);
        }
    }

    public boolean isLeaf() {
        if (LeftChild == null && RightChild == null)
            return true;
        else if (LeftChild != null && RightChild != null)
            return false;
        System.err.println("Broken tree, node has one child " + NodeName);
        return false;
    }
}
