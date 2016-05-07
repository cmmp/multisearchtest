package multisearchtest;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.classifiers.meta.multisearch.DefaultSearch;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.setupgenerator.MathParameter;

public class MultisearchTest {

    private static Random rng = new Random(42);

    public void start() throws Exception {
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(this.getClass().getResourceAsStream("/UCI/breast-cancer.arff")));
        Instances data = new Instances(reader);
        reader.close();

        System.out.println(data);

        data.setClassIndex(data.numAttributes() - 1);

        J48 j48 = new J48();

        MultiSearch multi = new MultiSearch();
        multi.setClassifier(j48);
        SelectedTag tag = new SelectedTag(DefaultEvaluationMetrics.EVALUATION_AUC,
                new DefaultEvaluationMetrics().getTags());
        multi.setEvaluation(tag);

        MathParameter[] params = new MathParameter[2];

        MathParameter conf = new MathParameter();
        conf.setProperty("confidenceFactor");
        conf.setMin(0.1);
        conf.setMax(0.5);
        conf.setStep(0.05);
        conf.setBase(10);
        conf.setExpression("I");
        params[0] = conf;

        MathParameter minNumObj = new MathParameter();
        minNumObj.setProperty("minNumObj");
        minNumObj.setMin(2);
        minNumObj.setMax(10);
        minNumObj.setStep(1);
        minNumObj.setBase(10);
        minNumObj.setExpression("I");
        params[1] = minNumObj;

        multi.setSearchParameters(params);

        multi.setDebug(true);

        multi.setSeed(rng.nextInt());

        multi.setAlgorithm(new DefaultSearch());
        multi.buildClassifier(data);

        System.out.println(multi.getBestClassifier());

        // Point<Object> bestValues = multi.getBestValues();

        // double confFactor = (Double) multi.getValues().getValue(0);
        // double minObj = (Double) multi.getValues().getValue(1);

        // System.out.println("I found confFactor = " + confFactor + " and minNumObj = " + minObj);

        // J48 bestClassifier = new J48();
        // bestClassifier.setConfidenceFactor((float) confFactor);
        // bestClassifier.setMinNumObj((int) minObj);
        // bestClassifier.setSeed(rng.nextInt());
        //
        // bestClassifier.buildClassifier(data);
        // return bestClassifier;

    }

    public static void main(String[] args) throws Exception {
        new MultisearchTest().start();
    }

}
