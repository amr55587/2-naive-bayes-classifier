using System;
using M = Microsoft.ML.Probabilistic.Models;
using D = Microsoft.ML.Probabilistic.Distributions;
using A = Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {
            /********* arguments **********/
            string dataDir = args[0];
            string datasetFilename = args[1];
            /******************************/

            /************ data ************/
            string[] lines = File.ReadAllLines(dataDir+datasetFilename);
            int numSamples = lines.Length;
            double[] x0Data = new double[numSamples];
            double[] x1Data = new double[numSamples];
            bool[] yData = new bool[numSamples];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] strArray = lines[i].Split('|');
                double[] doubleArray = Array.ConvertAll<string, double>(strArray, Convert.ToDouble);
                
                x0Data[i] = doubleArray[0];
                x1Data[i] = doubleArray[1];

                if (doubleArray[2] == 1)
                {
                    yData[i] = true;
                }
                else
                {
                    yData[i] = false;
                }
            }
            /********************************/

            /********* model setup **********/
            Range n = new Range(numSamples);

            M.VariableArray<bool> y = M.Variable.Array<bool>(n);

            M.VariableArray<double> x0 = M.Variable.Array<double>(n);
            M.VariableArray<double> x1 = M.Variable.Array<double>(n);

            double variance = 1.0;

            M.Variable<double> x0c0Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x0c1Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x1c0Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x1c1Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);

            M.Variable<double> cPrior = M.Variable.Beta(1, 1);

            using (M.Variable.ForEach(n))
            {
                y[n] = M.Variable.Bernoulli(cPrior);

                using (M.Variable.IfNot(y[n]))
                {
                    x0[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x0c0Mean, variance));
                    x1[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x1c0Mean, variance));
                }
                using (M.Variable.If(y[n]))
                {
                    x0[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x0c1Mean, variance));
                    x1[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x1c1Mean, variance));
                }
            }
            /********************************/

            /********* observations *********/
            x0.ObservedValue = x0Data;
            x1.ObservedValue = x1Data;
            y.ObservedValue = yData;
            /********************************/

            /******* inference engine *******/
            M.InferenceEngine engine = new M.InferenceEngine(new A.ExpectationPropagation());
            // engine.ShowFactorGraph = true;
            /********************************/

            /********** posteriors **********/
            D.Gaussian postx0c0Mean = engine.Infer<D.Gaussian>(x0c0Mean);
            D.Gaussian postx0c1Mean = engine.Infer<D.Gaussian>(x0c1Mean);
            D.Gaussian postx1c0Mean = engine.Infer<D.Gaussian>(x1c0Mean);
            D.Gaussian postx1c1Mean = engine.Infer<D.Gaussian>(x1c1Mean);
            D.Beta postClass = engine.Infer<D.Beta>(cPrior);
            /********************************/

            /********** print outs **********/
            Console.WriteLine("Posterior class: {0}", postClass);
            Console.WriteLine("Posterior class0 means: {0} {1}", postx0c0Mean, postx1c0Mean);
            Console.WriteLine("Posterior class1 means: {0} {1}", postx0c1Mean, postx1c1Mean);
            /********************************/

            /***** creating results.csv *****/
            var results = new StringBuilder();
            results.AppendLine("classPost|meanPost0|meanPost1");
            var line = string.Format("{0}|{1}|{2}", 1-postClass.GetMean(), postx0c0Mean.GetMean(), postx1c0Mean.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("{0}|{1}|{2}", postClass.GetMean(), postx0c1Mean.GetMean(), postx1c1Mean.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            File.WriteAllText(dataDir + "results.csv", results.ToString());
            /*********************************/
        }
    }
}
