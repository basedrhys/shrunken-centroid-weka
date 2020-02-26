package weka.classifiers.lazy;

import weka.core.DenseInstance;
import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

class Centroid implements Serializable  {
    private static final long serialVersionUID = 0;

    // Instance to hold the values
    private Instance m_inst;

    // Instance to hold the values of the shrunken version
    private Instance m_shrunkenInst;

    // Keep a list of instances for this class
    private List<Instance> m_instances;

    private int m_classIndex;

    public Centroid(int numAttributes, int classAttribute) {
        m_inst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_shrunkenInst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_instances = new ArrayList<>();
        m_classIndex = classAttribute;
    }

    public void setValue(int i, double v) {
        m_inst.setValue(i, v);
    }

    public void setShrunkenValue(int i, double v) { m_shrunkenInst.setValue(i, v);}

    public double getValue(int i) {
        return m_inst.value(i);
    }

    /**
     * Calculates how shrunken many attributes are non-zero
     * @param globalCentroid The global centroid to compare against
     * @return Number of attributes that are non-zero
     */
    public int getNonZeroShrunkenAttributes(Centroid globalCentroid) {
        int count = 0;
        for (int i = 0; i < m_shrunkenInst.numAttributes(); i++) {
            if (i != m_classIndex) {
                double val = Math.abs(m_shrunkenInst.value(i) - globalCentroid.getValue(i));
                if (val > 0) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Add all attribute values from this instance to the centroid
     * @param inst instance to add
     */
    public void addInstance(Instance inst) {
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            if (i != m_classIndex) {
                double newVal = m_inst.value(i) + inst.value(i);
                m_inst.setValue(i, newVal);
            }
        }
        m_instances.add(inst);
    }

    public Instance getInstance() {
        return m_inst;
    }

    public List<Instance> getInstances() {
        return m_instances;
    }

    public int numAttributes() { return m_inst.numAttributes(); }

    /**
     * @param instance Instance to compare against
     * @param attributeI Attribute index we're comparing
     * @return Difference between parameter instance and this centroid instance, for the given index
     */
    public double getDifferenceFromInstanceAttribute(Instance instance, int attributeI) {
        return instance.value(attributeI) - m_inst.value(attributeI);
    }

    /**
     * @param instance Instance to compare against
     * @param attributeI Attribute index we're comparing
     * @param useShrunken Use the shrunken centroid to compare
     * @return
     */
    public double getDifferenceFromInstanceAttribute(Instance instance, int attributeI, boolean useShrunken) {
        if (useShrunken)
            return instance.value(attributeI) - m_shrunkenInst.value(attributeI);
        else
            return getDifferenceFromInstanceAttribute(instance, attributeI);
    }

    /**
     * Calculates euclidean distance between this centroid and the given instance
     * @param instance Instance to measure against
     * @return
     */
    public double getDistanceFromInstance(Instance instance) {
        double dist = 0;
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            if (i != m_classIndex){
                double d = instance.value(i) - m_inst.value(i);
                dist += d * d;
            }
        }
        return Math.sqrt(dist);
    }

    /**
     * Average all attribute values.
     * 
     * Should be called after all instances in this class have been added to this centroid,
     * to find the final centroid location
     */
    public void averageValues() {
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double newVal = m_inst.value(i) / m_instances.size();
            m_inst.setValue(i, newVal);
        }
    }

    public String toString() {
        return String.format("Center = %s, Number of Instances = %d", m_inst.toString(), m_instances.size());
    }

}