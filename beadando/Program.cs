using CNTK;

namespace beadando
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DeviceDescriptor.TrySetDefaultDevice(DeviceDescriptor.CPUDevice);

            DataSet traindata = new DataSet(@"..\..\..\..\train.csv");
            BatchTraining app = new BatchTraining(traindata);

            app.Train();
            Console.ReadLine();
        }
    }
}
