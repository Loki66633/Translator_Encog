import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.lma.LevenbergMarquardtTraining;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;

import java.util.*;


public class Main {

    static final int BROJ_JEZIKA = 6;
    static final int BROJ_RIJECI =  10;

    public static final String noc="noc";
    public static final String kruh="kruh";
    public static final String tjestenina="tjestenina";
    public static final String jaja="jaja";
    public static final String boja="boja";
    public static final String sok="sok";
    public static final String riba="riba";
    public static final String zelena="zelena";
    public static final String piletina="piletina";
    public static final String jabuka="jabuka";

    public static final String[] listaPrijevoda = new String[]{noc, kruh, tjestenina, jaja, boja, sok, riba, zelena, piletina, jabuka};

    public static final String[] listaFrancuski = new String[]{"nuit", "pain", "pates", "oeuf", "couleur", "jus", "poisson", "vert", "pulet", "pomme"};
    public static final String[] listaSpanjolski = new String[]{"noche", "pan", "pasta", "huevo", "color", "jugo", "pescado", "verde", "pollo", "manzana"};
    public static final String[] listaPortugalski = new String[]{"noite", "pao", "massa", "ovo", "cor", "suco", "peicse", "verde", "frango", "maca"};
    public static final String[] listaTalijanski = new String[]{"notte", "pane", "pasta", "uovo", "colore", "succo", "pesce", "verde", "pollo", "mela"};
    public static final String[] listaEngleski = new String[]{"night", "bread", "pasta", "egg", "colour", "juice", "fish", "green", "chicken", "apple"};
    public static final String[] listaNjemacki = new String[]{"nacht", "brot", "pasta", "ei", "farbe", "saft", "fisch", "grun", "huhn", "apfel"};


    static final double N = listaPrijevoda.length;


    public static double[][] XOR_INPUT = {
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.4, 0.0, 0.0, 0.25, 0.0, 0.25},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.2, 0.2, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.25, 0.0, 0.25, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3333333333333333, 0.0, 0.0, 0.3333333333333333, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.25, 0.25, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.25, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.0, 0.25, 0.0, 0.25, 0.25},
            {1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.2},
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.6666666666666666, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.5, 0.25},
            {1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.3, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.0, 0.5, 0.5, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.7, 0.0, 0.14285714285714285, 0.0, 0.14285714285714285, 0.2857142857142857},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.6, 0.0, 0.16666666666666666, 0.0, 0.3333333333333333, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.3333333333333333, 0.16666666666666666},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333},
            {1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.25, 0.25},
            {1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.25, 0.25},
            {1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.2, 0.2},
            {1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.2, 0.0, 0.2},
            {1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.4, 0.25, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.14285714285714285, 0.2857142857142857, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.14285714285714285, 0.14285714285714285, 0.0, 0.14285714285714285, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.3333333333333333, 0.16666666666666666, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.4, 0.0, 0.0, 0.25, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.4, 0.0, 0.25, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.25},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.0, 0.2},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0},
            {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.6, 0.16666666666666666, 0.0, 0.0, 0.16666666666666666, 0.0},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.7, 0.0, 0.14285714285714285, 0.14285714285714285, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.25},
            {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.7, 0.42857142857142855, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4, 0.25, 0.25, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0}
    };

    public static double[][] NAKON_TRENINGA = {
            {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.4, 0.0, 0.25, 0.0, 0.25, 0.0}
    };

    /**
     * The ideal data necessary for XOR.
     */
    public static double[][] XOR_IDEAL = {


            { 1,0,0,0,0,0,0,0,0,0 }, //NOC
            { 1,0,0,0,0,0,0,0,0,0 }, //NOC
            { 1,0,0,0,0,0,0,0,0,0 }, //NOC
            { 1,0,0,0,0,0,0,0,0,0 }, //NOC
            { 1,0,0,0,0,0,0,0,0,0 }, //NOC
            { 1,0,0,0,0,0,0,0,0,0 }, //NOC

            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH
            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH
            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH
            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH
            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH
            { 0,1,0,0,0,0,0,0,0,0 }, //KRUH

            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA
            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA
            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA
            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA
            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA
            { 0,0,1,0,0,0,0,0,0,0 }, //TJESTENINA

            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA
            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA
            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA
            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA
            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA
            { 0,0,0,1,0,0,0,0,0,0 }, //JAJA

            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA
            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA
            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA
            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA
            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA
            { 0,0,0,0,1,0,0,0,0,0 }, //BOJA

            { 0,0,0,0,0,1,0,0,0,0 }, //SOK
            { 0,0,0,0,0,1,0,0,0,0 }, //SOK
            { 0,0,0,0,0,1,0,0,0,0 }, //SOK
            { 0,0,0,0,0,1,0,0,0,0 }, //SOK
            { 0,0,0,0,0,1,0,0,0,0 }, //SOK
            { 0,0,0,0,0,1,0,0,0,0 }, //SOK

            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA
            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA
            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA
            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA
            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA
            { 0,0,0,0,0,0,1,0,0,0 }, //RIBA

            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA
            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA
            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA
            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA
            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA
            { 0,0,0,0,0,0,0,1,0,0 }, //ZELENA

            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA
            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA
            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA
            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA
            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA
            { 0,0,0,0,0,0,0,0,1,0 }, //PILETINA

            { 0,0,0,0,0,0,0,0,0,1 },  //JABUKA
            { 0,0,0,0,0,0,0,0,0,1 },  //JABUKA
            { 0,0,0,0,0,0,0,0,0,1 },  //JABUKA
            { 0,0,0,0,0,0,0,0,0,1 },  //JABUKA
            { 0,0,0,0,0,0,0,0,0,1 },  //JABUKA
            { 0,0,0,0,0,0,0,0,0,1 }  //JABUKA
    };

    public static void main(String[] args) {
        // create a neural network, without using a factory
        //popuniStranePrijevode();
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, false   , 13));
        network.addLayer(new BasicLayer(new ActivationSteepenedSigmoid(), false, 100));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 10));
        network.getStructure().finalizeStructure();
        network.reset();

        // create training data
        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        // train the neural network
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while (train.getError() > 0.001);
        train.finishTraining();

        double correct = 0;

        // test the neural network
        System.out.println("Neural Network Results:");
        for (MLDataPair pair : trainingSet) {
            final MLData output = network.compute(pair.getInput());
            System.out.println("INPUT:");
            for (int i = 0; i < pair.getInput().getData().length; i++) {
                System.out.print(pair.getInput().getData(i) + ",");
                if (i == 6 || i == 7) {
                    System.out.print(" -- ");
                }
            }
            System.out.println();

            System.out.println("\nIDEAL OUTPUT:");
            for (int i = 0; i < pair.getIdeal().getData().length; i++) {
                System.out.print(pair.getIdeal().getData(i) + ",");
            }

            System.out.println();

            System.out.println("\nACTUAL OUTPUT:");
            for (int i = 0; i < output.getData().length; i++) {
                System.out.printf("%.3f", output.getData(i));
                System.out.print(",");
            }
            System.out.println("\n");
            System.out.println("Ispravno: " + convertIdealToString(pair.getIdealArray()));
            System.out.println("Prevedeno: " + convertIdealToStringAndPrintSigurnost(output.getData()));
            if (convertIdealToString(output.getData()).equals(convertIdealToString(pair.getIdeal().getData()))) {
                correct++;
            } else {
                System.out.println("GREŠKA! MREŽA NIJE SIGURNA ŠTO JE RJEŠENJE");
            }
            System.out.println("------------------------------------------------------------------------------------");

        }

        System.out.println("Ukupno točnih prijevoda: " + (int) correct + "/60");
        System.out.println("Treniranje mreže je gotovo!\nUspješnost: " + (correct / 60) * 100 + "%");
        System.out.println("Ukupno epoha: " + epoch);

        System.out.println("------------------------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------------------------");


        Scanner scan = new Scanner(System.in);
        System.out.print("Unesite rijec koju želite prevesti na hrvatski: ");
        String unos = scan.nextLine();

        ArrayList<Double> konvertiraniUnos = rijecUInput(unos);
        double[] arr = konvertiraniUnos.stream().mapToDouble(Double::doubleValue).toArray();
        NAKON_TRENINGA[0] = arr;

        MLDataSet trainingSet2 = new BasicMLDataSet(NAKON_TRENINGA, XOR_IDEAL);


        for (MLDataPair pair : trainingSet2) {
            final MLData output = network.compute(pair.getInput());
            System.out.println("\nOUTPUT:");
            for (int i = 0; i < output.getData().length; i++) {
                System.out.printf("%.3f", output.getData(i));
                System.out.print(",");
            }
            System.out.println("\n");
            System.out.print("Prijevod riječi " + unos + " je: ");
            System.out.println(convertIdealToString(output.getData()));
            printSigurnost(output.getData());
            System.out.println("------------------------------------------------------------------------------------");

        }
        Encog.getInstance().shutdown();
    }













    public static String convertIdealToString(double[] polje){
        String prijevod="";
        double sigurnost=0;
        int index = getIndexOfLargest(polje);

        switch (index) {
            case 0 -> prijevod = noc;
            case 1 -> prijevod = kruh;
            case 2 -> prijevod = tjestenina;
            case 3 -> prijevod = jaja;
            case 4 -> prijevod = boja;
            case 5 -> prijevod = sok;
            case 6 -> prijevod = riba;
            case 7 -> prijevod = zelena;
            case 8 -> prijevod = piletina;
            case 9 -> prijevod = jabuka;
            default -> prijevod = "Greška";
        }
        return prijevod;
    }
    public static String convertIdealToStringAndPrintSigurnost(double[] polje){
        String prijevod="";
        double sigurnost=0;
        int index = getIndexOfLargest(polje);
        switch (index) {
            case 0 -> prijevod = noc;
            case 1 -> prijevod = kruh;
            case 2 -> prijevod = tjestenina;
            case 3 -> prijevod = jaja;
            case 4 -> prijevod = boja;
            case 5 -> prijevod = sok;
            case 6 -> prijevod = riba;
            case 7 -> prijevod = zelena;
            case 8 -> prijevod = piletina;
            case 9 -> prijevod = jabuka;
            default -> prijevod = "Greška";
        }
        System.out.println("Sigurnost: " + polje[index]*100 + "%");
        return prijevod;
    }
    public static void printSigurnost(double[] polje){
        String prijevod="";
        double sigurnost=0;
        int index = getIndexOfLargest(polje);
        switch (index) {
            case 0 -> prijevod = noc;
            case 1 -> prijevod = kruh;
            case 2 -> prijevod = tjestenina;
            case 3 -> prijevod = jaja;
            case 4 -> prijevod = boja;
            case 5 -> prijevod = sok;
            case 6 -> prijevod = riba;
            case 7 -> prijevod = zelena;
            case 8 -> prijevod = piletina;
            case 9 -> prijevod = jabuka;
            default -> prijevod = "Greška";
        }
        System.out.println("Sigurnost: " + polje[index]*100 + "%");
    }

    public static void popuniStranePrijevode(){
        int counter = 0;
        for(int j=0;j<BROJ_RIJECI;j++){
            for(int i=0;i<BROJ_JEZIKA;i++){
                counter++;

                ArrayList<Double> polje = new ArrayList<>();

                if(i==0){ //FRANCUSKI
                    String rijec = listaFrancuski[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);

                }
                else if(i==1){ //SPANJOLSKI
                    String rijec = listaSpanjolski[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);
                }
                else if(i==2){ //PORTUGALSKI
                    String rijec = listaPortugalski[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);
                }
                else if(i==3){ //TALIJANSKI
                    String rijec = listaTalijanski[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);
                }
                else if(i==4){ //ENGLESKI
                    String rijec = listaEngleski[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);
                }
                else if(i==5){ //NJEMACKI
                    String rijec = listaNjemacki[j];
                    ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

                    double duljinaRijeci = rijec.length()/N;
                    Double[] samoglasnici = new Double[5];
                    Arrays.fill(samoglasnici,0.0);
                    for(int k=0;k<rijec.length();k++) {
                        if(rijec.charAt(k)=='a'){
                            samoglasnici[0]++;
                        }
                        if(rijec.charAt(k)=='e'){
                            samoglasnici[1]++;
                        }
                        if(rijec.charAt(k)=='i'){
                            samoglasnici[2]++;
                        }
                        if(rijec.charAt(k)=='o'){
                            samoglasnici[3]++;
                        }
                        if(rijec.charAt(k)=='u'){
                            samoglasnici[4]++;
                        }
                    }
                    for(int l=0;l<samoglasnici.length;l++){
                        if(samoglasnici[l]!=0.0){
                            samoglasnici[l] = samoglasnici[l]/rijec.length();
                        }

                    }
                    List<Double> samog = Arrays.asList(samoglasnici);
                    polje.addAll(prvoSlovo);
                    polje.add(duljinaRijeci);
                    polje.addAll(samog);
                }

                System.out.println(polje);

            }
        }

    } //Pokrenemo da nam da ispis koji onda koristimo za unos

    public static ArrayList<Double> prvoSlovoToBin(String rijec){
        char slovo = rijec.charAt(0);
        String prvoSlovo = Integer.toBinaryString(slovo);
        String[] prvoSlovoArr = prvoSlovo.split("");
        ArrayList<Double> arr = new ArrayList<>(prvoSlovoArr.length);
        for(int i=0;i<prvoSlovoArr.length;i++){
            arr.add((double) Integer.parseInt(prvoSlovoArr[i])) ;
        }
        return arr;
    }
    public static int getIndexOfLargest( double[] array ) {
        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest;
    }


    public static ArrayList<Double> rijecUInput(String rijec){
        // {1,1,0,1,1,1,0 ,0.4, 0,0,0,0,0.25},
        rijec = rijec.toLowerCase(Locale.ROOT);
        ArrayList<Double> polje = new ArrayList<>();
        ArrayList<Double> prvoSlovo = prvoSlovoToBin(rijec);

        double duljinaRijeci = rijec.length()/10.0;
        Double[] samoglasnici = new Double[5];
        Arrays.fill(samoglasnici,0.0);
        for(int k=0;k<rijec.length();k++) {
            if(rijec.charAt(k)=='a'){
                samoglasnici[0]++;
            }
            if(rijec.charAt(k)=='e'){
                samoglasnici[1]++;
            }
            if(rijec.charAt(k)=='i'){
                samoglasnici[2]++;
            }
            if(rijec.charAt(k)=='o'){
                samoglasnici[3]++;
            }
            if(rijec.charAt(k)=='u'){
                samoglasnici[4]++;
            }
        }
        for(int l=0;l<samoglasnici.length;l++){
            if(samoglasnici[l]!=0.0){
                samoglasnici[l] = samoglasnici[l]/rijec.length();
            }

        }
        List<Double> samog = Arrays.asList(samoglasnici);
        polje.addAll(prvoSlovo);
        polje.add(duljinaRijeci);
        polje.addAll(samog);


        System.out.println(polje);
        return polje;
    }
}