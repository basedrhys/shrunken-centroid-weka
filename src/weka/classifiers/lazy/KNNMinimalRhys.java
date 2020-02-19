package weka.classifiers.lazy;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.classifiers.AbstractClassifier;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class KNNMinimalRhys extends AbstractClassifier {

    protected int m_K = 1;

    protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

    @OptionMetadata(displayName = "number of neighbours", description = "Number of neighbours to use (default = 1).",
            commandLineParamName = "K", commandLineParamSynopsis = "-K <int>", displayOrder = 1)
    public void setK(int k) {
        m_K = k;
    }

    public int getK() {
        return m_K;
    }

    public void buildClassifier(Instances trainingData) throws Exception {

        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        m_NNSearch.setInstances(trainingData);
    }

    public double[] distributionForInstance(Instance testInstance) throws Exception {

        m_NNSearch.addInstanceInfo(testInstance);

        Instances neighbours = m_NNSearch.kNearestNeighbours(testInstance, m_K);

        double[] dist = new double[testInstance.numClasses()];
        for (Instance neighbour : neighbours) {
            if (testInstance.classAttribute().isNominal()) {
                dist[(int)neighbour.classValue()] += 1.0 / neighbours.numInstances();
            } else {
                dist[0] += neighbour.classValue() / neighbours.numInstances();
            }
        }
        return dist;
    }
}
