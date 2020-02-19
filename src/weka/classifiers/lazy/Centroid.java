package weka.classifiers.lazy;

import weka.core.DenseInstance;
import weka.core.Instance;

class Centroid {
    // Instance to hold the values TODO should we use a double[] instead of Instance?
    Instance m_inst;

    // Number of instances for this class centroid
    int m_numInstances;

    public Centroid(int numAttributes) {
        m_inst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_numInstances = 0;
    }

    public Instance getInst() {
        return m_inst;
    }

    public void setInst(Instance m_inst) {
        this.m_inst = m_inst;
    }

    public int getNumInstances() {
        return m_numInstances;
    }

    public void setNumInstances(int m_numInstances) {
        this.m_numInstances = m_numInstances;
    }

    public double getAttributeValue(int i) {
        return this.m_inst.value(i);
    }

    public void setAttributeValue(int i, double v) {
        this.m_inst.setValue(i, v);
    }

    public void addInstance(Instance inst) {
        // Add all attribute values from this instance to both the global centroid
        // and the appropriate class centroid
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double newVal = m_inst.value(i) + inst.value(i);
            m_inst.setValue(i, newVal);
        }
        m_numInstances++;
    }

    public void averageValues() {
        // Average all attribute values
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double newVal = m_inst.value(i) / m_numInstances;
            m_inst.setValue(i, newVal);
        }
    }

    public String toString() {
        return String.format("Center = %s, Number of Instances = %d", m_inst.toString(), m_numInstances);
    }

}