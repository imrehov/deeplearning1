using CNTK;

namespace cntk_basic_nn
{


    class AdvancedTraining
    {
        const int inputSize = 4;
        const int hiddenNeuronCount = 3;
        const int outputSize = 1;

        readonly Variable x;
        readonly Function y;

        public AdvancedTraining()
        {
            x = Variable.InputVariable(new int[] { inputSize }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, inputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter w2 = new Parameter(new int[] { outputSize, hiddenNeuronCount}, DataType.Float, CNTKLib.GlorotNormalInitializer());
            y = CNTKLib.Sigmoid(CNTKLib.Times(w2, CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w1, x), b))));
        }

        public void Train(string[] trainData)
        {
            int n = trainData.Length;

            Variable yt = Variable.InputVariable(new int[] { 1, outputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner});

            for (int i = 1; i <=1000; i++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                foreach (string line in trainData)
                {
                    float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        {x, null}
                    };

                }
            }
        }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }
    }
}
