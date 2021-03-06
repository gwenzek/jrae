package rae;

import classify.ClassifierTheta;
import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.Scanner;

public class FineTunableTheta extends Theta {

    DoubleMatrix Wcat, bcat;
    int catSize;
    private static final long serialVersionUID = 752647956355547L;

    public FineTunableTheta(double[] iTheta, int hiddenSize, int visibleSize, int CatSize, int dictionaryLength) {
        super();
        this.hiddenSize = hiddenSize;
        this.visibleSize = visibleSize;
        this.dictionaryLength = dictionaryLength;
        this.catSize = CatSize;
        fixIndices();
        DoubleMatrix Full = new DoubleMatrix(iTheta);

        W1 = Full.getRowRange(Wbegins[0], Wends[0] + 1, 0).reshape(hiddenSize, visibleSize);
        W2 = Full.getRowRange(Wbegins[1], Wends[1] + 1, 0).reshape(hiddenSize, visibleSize);
        W3 = Full.getRowRange(Wbegins[2], Wends[2] + 1, 0).reshape(visibleSize, hiddenSize);
        W4 = Full.getRowRange(Wbegins[3], Wends[3] + 1, 0).reshape(visibleSize, hiddenSize);
        We = Full.getRowRange(Wbegins[4], Wends[4] + 1, 0).reshape(hiddenSize, dictionaryLength);

        b1 = Full.getRowRange(bbegins[0], bends[0] + 1, 0).reshape(hiddenSize, 1);
        b2 = Full.getRowRange(bbegins[1], bends[1] + 1, 0).reshape(visibleSize, 1);
        b3 = Full.getRowRange(bbegins[2], bends[2] + 1, 0).reshape(visibleSize, 1);


        Wcat = Full.getRowRange(Wbegins[5], Wends[5] + 1, 0).reshape(CatSize, hiddenSize);
        bcat = Full.getRowRange(bbegins[5], bends[5] + 1, 0).reshape(CatSize, 1);

        theta = new double[getThetaSize()];
        flatten(theta);
    }

    public FineTunableTheta(FineTunableTheta orig) {
        super(orig);
        catSize = orig.catSize;
        Wcat = orig.Wcat.dup();
        bcat = orig.bcat.dup();
    }

    public FineTunableTheta(int hiddenSize, int visibleSize, int catSize, int dictionaryLength, boolean random) {
        super(hiddenSize, visibleSize, catSize, dictionaryLength);

        this.catSize = catSize;
        if (random)
            InitializeMatrices();
        else
            InitializeMatricesToZeros();
        theta = new double[this.getThetaSize()];
        flatten(theta);
    }

    /**
     * Set the Ws and bs and populate theta
     */
    public FineTunableTheta(DoubleMatrix W1, DoubleMatrix W2,
                            DoubleMatrix W3, DoubleMatrix W4, DoubleMatrix Wcat,
                            DoubleMatrix We, DoubleMatrix b1, DoubleMatrix b2,
                            DoubleMatrix b3, DoubleMatrix bcat) {
        this.W1 = W1;
        this.W2 = W2;
        this.W3 = W3;
        this.W4 = W4;
        this.We = We;
        this.b1 = b1;
        this.b2 = b2;
        this.b3 = b3;
        this.Wcat = Wcat;
        this.bcat = bcat;

        hiddenSize = W1.rows;
        visibleSize = W1.columns;
        dictionaryLength = We.columns;
        catSize = bcat.rows;

        theta = new double[getThetaSize()];
        flatten(theta);
    }

    public void dump(String FileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(FileName);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(this);
        oos.flush();
        oos.close();
    }

    @Override
    public int getThetaSize() {
        return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength
                + hiddenSize + 2 * visibleSize + catSize * hiddenSize + catSize;
    }

    public int getNumCategories() {
        return catSize + 1;
    }

    public ClassifierTheta getClassifierParameters() {
        return new ClassifierTheta(Wcat.transpose(), bcat);
    }

    @Override
    protected void InitializeMatrices() {
        super.InitializeMatrices();
        Wcat = (DoubleMatrix.rand(catSize, hiddenSize).muli(2 * r1)).subi(r1);
        bcat = DoubleMatrix.zeros(catSize, 1);
    }

    @Override
    protected void InitializeMatricesToZeros() {
        super.InitializeMatricesToZeros();
        Wcat = DoubleMatrix.zeros(catSize, hiddenSize);
        bcat = DoubleMatrix.zeros(catSize, 1);
    }

    @Override
    protected void flatten(double[] Theta) {
        fixIndices();
        super.flatten(Theta);
        System.arraycopy(Wcat.toArray(), 0, Theta, Wbegins[5], catSize * hiddenSize);
        System.arraycopy(bcat.toArray(), 0, Theta, bbegins[5], catSize);
    }

    @Override
    protected void fixIndices() {
        super.fixIndices();
        Wbegins[5] = bends[2] + 1;
        Wends[5] = Wbegins[5] + catSize * hiddenSize - 1;    //Wcat
        bbegins[5] = Wends[5] + 1;
        bends[5] = bbegins[5] + catSize - 1;                //bcat

//		for(int i=0; i<=5; i++)
//			System.out.println (Wbegins[i] + " " + Wends[i]);
//		System.out.println ("----");
//		for(int i=0; i<=5; i++)
//			System.out.println (bbegins[i] + " " + bends[i]);		
//		System.out.println ("----");
//		System.out.println ("----");
    }

    public void dumpAsCSV(String dir) {
        //we = word matrix
        try {
            FileWriter out = new FileWriter(dir + "/We.csv");
            out.write("" + We.getColumns() + ";" + We.getRows() + "\n");
            for (int i = 0; i < We.getColumns(); i++) {
                for (int j = 0; j < We.getRows(); j++) {
                    out.write("" + We.get(i, j) + ";");
                }
                out.write("\n");
            }
            out.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }

        try {
            //Wcat = sentiment classification Matrix
            FileWriter out = new FileWriter(dir + "/Wcat.csv");
            out.write("" + Wcat.getRows() + ";" + (Wcat.getColumns()+1) + "\n");
            for (int i = 0; i < Wcat.getRows(); i++) {
                out.write(""+bcat.get(i, 0)+";");
                for (int j = 0; j < Wcat.getColumns(); j++) {
                    out.write("" + Wcat.get(i, j) + ";");
                }
                out.write("\n");
            }
            out.close();

        } catch (IOException e) {
            System.err.println(e.getMessage());
        }

        // W1 and W2 are the matrix used to construct the representation of two words

        try {
            FileWriter out = new FileWriter(dir + "/W12.csv");
            out.write("" + (W1.getRows()+W2.getRows()) + ";" + (W1.getColumns()+1) + "\n");
            for (int i = 0; i < W1.getRows(); i++) {
                out.write(""+b1.get(i, 0)+";");
                for (int j = 0; j < W1.getColumns(); j++) {
                    out.write("" + W1.get(i, j) + ";");
                }
                for (int j = 0; j < W2.getColumns(); j++) {
                    out.write("" + W2.get(i, j) + ";");
                }
                out.write("\n");
            }
            out.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }

    public void dumpWords(String outFilename, String wordMapFilename) {
        try{
            Scanner wordMap = new Scanner(new FileReader(wordMapFilename));
            FileWriter out = new FileWriter(outFilename);
            String word;
            int index;
            double[] vector;
            while(wordMap.hasNext()){
                word = wordMap.next();
                index = wordMap.nextInt();
                out.write(word+" ");
                vector = We.getColumn(index).data;
                for(double d : vector)
                    out.write(""+d+" ");
                out.write("\n");
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
