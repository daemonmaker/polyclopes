using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Camera_Controller : MonoBehaviour
{
    public static bool start;
    public static int Cameras;
    // Start is called before the first frame update
    void Start()
    {
        start = false;
        Cameras = 1;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space) == true)
        {
            start = true;
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Application.LoadLevel("Cube");
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            Application.LoadLevel("Capsule");
        }
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            Application.LoadLevel("Cylinder");
        }
        if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            Application.LoadLevel("Lamp");
        }
        if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            Application.LoadLevel("Rectangle");
        }
        if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            Application.LoadLevel("Ramp");
        }
        if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            Application.LoadLevel("Spinner");
        }
        if (Input.GetKeyDown(KeyCode.Alpha8))
        {
            Application.LoadLevel("Columns");
        }
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            Application.LoadLevel("Ladder");
        }
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            Application.LoadLevel("Stairs");
        }
    }
}
