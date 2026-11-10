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
            //kiolvassuk a csv méretét
            int n = trainData.Length;

            //ki kell egészíteni a gráfot
            Variable yt = Variable.InputVariable(new int[] { 1, outputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            //mivel tanítunk
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
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
                        //normalizálás
                        {x, LoadInput(values[0], values[1], values[2], values[3])},
                        {yt, Value.CreateBatch(yt.Shape, new float[] {values[4]}, DeviceDescriptor.CPUDevice) }
                    };

                    var outputDataMap = new Dictionary<Variable, Value>() { { loss, null } };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);

                    sumLoss += trainer.PreviousMinibatchLossAverage();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage();
                }

                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}",i, sumLoss / n, sumEval / n));
            }
        }

        //
        public float Prediction(float age, float height, float weight, float salary)
        {
            var inputDataMap = new Dictionary<Variable, Value>() { { x, LoadInput(age, height, weight, salary) }  };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null} };

            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);

            return outputDataMap[y].GetDenseData<float>(y)[0][0];
        }

        //soronként betölt és normalizál
        Value LoadInput(float age, float height, float weight, float salary)
        { 
            float[] xstore = new float[inputSize];
            xstore[0] = age / 100;
            xstore[1] = height / 250;
            xstore[2] = weight / 150;
            xstore[3] = salary / 1500000;

            return Value.CreateBatch(x.Shape, xstore, DeviceDescriptor.CPUDevice);
        }
    }
    internal class Program
    {
        string[] trainData = File.ReadAllLines(@"..\data\HusbandEvaluation.txt");

        AdvancedTraining app = new AdvancedTraining();

        void Run()
        {
            app.Train(trainData);
            FileTest();
            ConsoleTest();

        }

        void FileTest()
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;

            foreach (string line in trainData)
            {
                float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                int good = (int)values[4];
                int pred = (int)Math.Round(app.Prediction(values[0], values[1], values[2], values[3]));

                if (pred == good)
                    if (pred == 1)
                        TP++;
                    else
                        TN++;
                else
                    if (pred == 1)
                        FP++;
                    else
                        FN++;
            }

            float accuracy = (float)(TP + TN) / (TP + FP + TN + FN);
            float precision = (float)TP / (TP + FP);
            float sensitivity = (float)TP / (TP + FN);
            float F1 = 2 * (precision * sensitivity) / (precision + sensitivity);

            Console.WriteLine(String.Format(
                "True positive:\t{0}" +
                "True negative:\t{1}" +
                "False positive:\t{2}" +
                "False negative:\t{3}",
                TP, TN, FP, FN
                ));

            Console.WriteLine(String.Format(
                "Accuracy:\t{0}" +
                "Precision:\t{1}" +
                "Sensitivity:\t{2}" +
                "F1:\t{3}",
                accuracy, precision, sensitivity, F1
                ));
        }

        void ConsoleTest()
        {
            Console.WriteLine("Age:");
            float age = float.Parse(Console.ReadLine());

            Console.WriteLine("Height:");
            float height = float.Parse(Console.ReadLine());

            Console.WriteLine("Weight:");
            float weight = float.Parse(Console.ReadLine());

            Console.WriteLine("Salary:");
            float salary = float.Parse(Console.ReadLine());

            Console.WriteLine("Prediction" + app.Prediction(age, height, weight, salary));
        }
        static void Main(string[] args)
        {

            new Program().Run();
        }
    }
}
