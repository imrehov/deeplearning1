namespace cntk_02;

class Program
{
    static float myFunc(float x, float a)
    {
        return (x - a) * (x - a);
    }

    static float myFuncD(float x, float a)
    {
        return 2 * (x - a);
    }

    static void gd01()
    {
        float x = 67.0f;

        const float a = 3.0f;

        const float alpha = 2.0f;

        for (int i = 0; i < 50; i++)
        {
            float fx = myFunc(x, a);
            float dfx = myFuncD(x, a);

            System.Console.WriteLine($"i: {i} x: {x} fx: {fx:F7} dfx: {dfx:F7}\n");
            x -= alpha * dfx;

        }
    }
    static void Main(string[] args)
    {
        gd_02 a = new gd_02(2.0f, 3.0f, 4.0f, 4.0f);
    }

}
