using System.ComponentModel;

namespace cntk_02;

public class gd_02
{
    private float alpha = 0.12f;

    public gd_02(float x, float y, float a, float b)
    {
        float myx = x;
        float myy = y;

        for (int i = 0; i < 50; i++)
        {
            float fx = xminusa(myx, a);
            float fy = yminusb(myy, b);

            float dfx = partialXminusa(myx, a);
            float dfy = partialYminusB(myy, b);

            System.Console.WriteLine("i: {0}\tx: {1}\ty: {2}\ttotal: {3:F10}", i, myx, myy, myFunc(myx, myy, a, b));

            myx -= alpha * dfx;
            myy -= alpha * dfy;
        }
    }

    private float myFunc(float x, float y, float a, float b)
    {
        return xminusa(x, a) + yminusb(y, b);
    }
    private float yminusb(float y, float b)
    {
        return (y - b) * (y - b);
    }
    
    private float partialYminusB(float y, float b)
    {
        return 2 * (y - b);
    }

    private float xminusa(float x, float a)
    {
        return (x - a) * (x - a);
    }

    private float partialXminusa(float x, float a)
    {
        return 2 * (x - a);
    }
}
