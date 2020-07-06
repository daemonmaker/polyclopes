using UnityEngine;
using System.Collections;
using System.IO;

public class HiResScreenShots : MonoBehaviour
{
    public int resWidth = 2550;
    public int resHeight = 3300;
    public static int zone = 12;
    public static int zone2 = 0;

    private bool takeHiResShot = false;

    public static string ScreenShotName(int width, int height)
    {
        return string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png",
                             Application.dataPath,
                             width, height,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }

    public void TakeHiResShot()
    {
        takeHiResShot = true;
    }

    void LateUpdate()
    {
        if (Camera_Controller.start == true)
        {
            if (Camera_Controller.Cameras > 12)
            {
                Camera_Controller.start = false;
            }
            else
            {
                takeHiResShot = true; //Input.GetKeyDown("k");
            }
            //Camera_Controller.start = false;
        }
        if (takeHiResShot)
        {
            RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
            GetComponent<Camera>().targetTexture = rt;
            Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
            GetComponent<Camera>().Render();
            RenderTexture.active = rt;
            screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
            GetComponent<Camera>().targetTexture = null;
            RenderTexture.active = null; // JC: added to avoid errors
            Destroy(rt);
            byte[] bytes = screenShot.EncodeToPNG();
            //string filename = ScreenShotName(resWidth, resHeight);
            File.WriteAllBytes(@"C:\Users\User\Pictures\Coding Pictures\Dad's Pictures\" + "thing"+ Camera_Controller.Cameras + ".png", bytes);
            //Debug.Log(string.Format("Took screenshot to: {0}", filename));
            Camera_Controller.Cameras += 1;
            takeHiResShot = false;
        }
    }
}