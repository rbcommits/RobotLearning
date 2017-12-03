using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class ArmController : MonoBehaviour
{
    public GameObject Robot;
    public GameObject[] joints;
    public GameObject[] links;
    public Vector3[] joint_rotation_constraints;
    public GameObject end_effector;
    public int[] link_length;
    private float animation_speed = 20;
    
    public InputField[] input_angles;
    private Rotate[] rotation_scripts;

    public GameObject cyl;

    // Use this for initialization
    void Start()
    {
        create_arm();
        setPythonShell();
        rotation_scripts = new Rotate[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            Rotate rotate_script = joints[i].GetComponent<Rotate>();
            rotate_script.rotation_constraint = joint_rotation_constraints[i];
            rotate_script.angle = 0;
            rotate_script.anim_speed = animation_speed;
            rotation_scripts[i] = rotate_script;

        }
    }
    public void testMoveCyl()
    {
        int a, b, c;
        int.TryParse(input_angles[0].text, out a);
        int.TryParse(input_angles[1].text, out b);
        int.TryParse(input_angles[2].text, out c);
        cyl.transform.forward = new Vector3(a, b, c);
    }
    void create_arm()
    {
        for (int i = 0; i < links.Length; i++)
        {
            float length = link_length[i];
            links[i].transform.localScale = new Vector3(1, length, 1);
            links[i].transform.localPosition = new Vector3(0, length, 0);
            if(i != 0)
            {
                joints[i].transform.localPosition = new Vector3(0, link_length[i - 1] * 2, 0);
            }
            
        }
        end_effector.transform.localPosition = new Vector3(0, link_length[5] * 2, 0);
    }
    // Update is called once per frame
    void Update()
    {

    }

    public void Moveto()
    {
        float[] angles = new float[input_angles.Length];

        float.TryParse(input_angles[0].text, out angles[0]);
        for (int i = 1; i < input_angles.Length; i++)
        {
            float.TryParse(input_angles[i].text, out angles[i]);
            angles[i] = Mathf.Clamp(angles[i], -90, 90);
        }

        for (int i = 0; i < joints.Length; i++)
        {
            rotation_scripts[i].angle = angles[i];
        }
    }

    string python = @"C:\ProgramData\Anaconda3\python.exe";
    string app = @"C:\Users\raagh\Desktop\forward_kin.py";
    string arguments = "";
    ProcessStartInfo myProcessStartInfo;
    Process myProcess;
    StreamReader myStreamReader;
    private void setPythonShell()
    {
        myProcessStartInfo = new ProcessStartInfo(python);
        myProcessStartInfo.UseShellExecute = false;
        myProcessStartInfo.RedirectStandardOutput = true;
        myProcessStartInfo.CreateNoWindow = true;
        myProcess= new Process();
    }
    public void GetWorldCoordinates()
    {
        for (int oo = 0; oo < 100; oo++)
        {
            Vector3[] angles = new Vector3[joints.Length];

            for (int i = 0; i < rotation_scripts.Length; i++)
            {
                angles[i] = rotation_scripts[i].GetRelativeAngle();
            }

            //Vector3[] linkpositions = GetCoordinate(float[] angles, Vector3[] linkpos, Vector3[] axes, int index)

            arguments = string.Format(" {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}",
                angles[0].magnitude, angles[1].magnitude, angles[2].magnitude, angles[3].magnitude, angles[4].magnitude, angles[5].magnitude, 0,
                link_length[0], link_length[1], link_length[2], link_length[3], link_length[4], link_length[5], 1
                );
            app = string.Concat(app, arguments);
            //UnityEngine.Debug.Log("arguments: " + app);



            myProcessStartInfo.Arguments = app;
            myProcess.StartInfo = myProcessStartInfo;
            myProcess.Start();

            myStreamReader = myProcess.StandardOutput;
            string result = myStreamReader.ReadToEnd();
            myProcess.WaitForExit();
            myProcess.Close();
            //UnityEngine.Debug.Log(result);
        }

        UnityEngine.Debug.LogError("Finished calling process");
        
    }


    private Vector3[] GetCoordinate(float[] angles, Vector3[] linkpos, Vector3[] axes, int index)
    {
        // Initialize
        Vector3 new_linkpos = linkpos[index];
        float q = angles[index];
        q = Mathf.Deg2Rad * (q); //convert to radiuns

        // Get Euler angles
        Vector3 eulerAngles = Quaternion.AngleAxis(q, axes[index]).eulerAngles;
        float alpha = eulerAngles.x; float beta = eulerAngles.y; float gama = eulerAngles.z;

        float[,] Rz = new float[,]
        {
            { Mathf.Cos(alpha), -Mathf.Sin(alpha), 0  },
            { Mathf.Sin(alpha),  Mathf.Cos(alpha),  0 },
            { 0, 0, 1  }
        };

        /*
        List <Vector3> Rz = new List<Vector3>();
        Rz.Add(new Vector3( Mathf.Cos(alpha), -Mathf.Sin(alpha), 0 ));
        Rz.Add(new Vector3( Mathf.Sin(alpha),  Mathf.Cos(alpha), 0 ));
        Rz.Add(new Vector3( 0, 0, 1 ));
        */

        float[,] Ry = new float[,]
        {
            { Mathf.Cos(beta), 0, Mathf.Sin(alpha)  },
            { 0, 1, 0 },
            { -Mathf.Sin(beta), 0, Mathf.Cos(beta)  }
        };

        /*
        List<Vector3> Ry = new List<Vector3>();
        Ry.Add(new Vector3( Mathf.Cos(beta), 0, Mathf.Sin(alpha) ));
        Ry.Add(new Vector3( 0, 1, 0 ));
        Ry.Add(new Vector3( -Mathf.Sin(beta), 0, Mathf.Cos(beta) ));
        */

        float[,] Rx = new float[,]
        {
            { 1, 0, 0  },
            { 0, Mathf.Cos(gama), -Mathf.Sin(gama) },
            { 0, Mathf.Sin(gama), Mathf.Cos(gama)  }
        };

        /*
        List<Vector3> Rx = new List<Vector3>();
        Rx.Add(new Vector3( 1, 0, 0 ));
        Rx.Add(new Vector3( 0, Mathf.Cos(gama), -Mathf.Sin(gama) ));
        Rx.Add(new Vector3( 0, Mathf.Sin(gama), Mathf.Cos(gama)  ));
        */

        float[,] Rzy = new float[,]
        {
            {   (Rz[0, 0] * Ry[0, 0]) + (Rz[0, 1] * Ry[1, 0]) + (Rz[0, 2] * Ry[2, 0]),    (Rz[0, 0] * Ry[0, 1]) + (Rz[0, 1] * Ry[1, 1]) + (Rz[0, 2] * Ry[2, 1]),    (Rz[0, 0] * Ry[0, 2]) + (Rz[0, 1] * Ry[1, 2]) + (Rz[0, 2] * Ry[2, 2]) },
            {   (Rz[1, 0] * Ry[0, 0]) + (Rz[1, 1] * Ry[1, 0]) + (Rz[1, 2] * Ry[2, 0]),    (Rz[1, 0] * Ry[0, 1]) + (Rz[1, 1] * Ry[1, 1]) + (Rz[1, 2] * Ry[2, 1]),    (Rz[1, 0] * Ry[0, 2]) + (Rz[1, 1] * Ry[1, 2]) + (Rz[1, 2] * Ry[2, 2]) },
            {   (Rz[2, 0] * Ry[0, 0]) + (Rz[2, 1] * Ry[1, 0]) + (Rz[2, 2] * Ry[2, 0]),    (Rz[2, 0] * Ry[0, 1]) + (Rz[2, 1] * Ry[1, 1]) + (Rz[2, 2] * Ry[2, 1]),    (Rz[2, 0] * Ry[0, 2]) + (Rz[2, 1] * Ry[1, 2]) + (Rz[2, 2] * Ry[2, 2]) }
        };

        float[,] R = new float[,]
        {
            {   (Rzy[0, 0] * Rx[0, 0]) + (Rzy[0, 1] * Rx[1, 0]) + (Rzy[0, 2] * Rx[2, 0]),    (Rzy[0, 0] * Rx[0, 1]) + (Rzy[0, 1] * Rx[1, 1]) + (Rzy[0, 2] * Rx[2, 1]),    (Rzy[0, 0] * Rx[0, 2]) + (Rzy[0, 1] * Rx[1, 2]) + (Rzy[0, 2] * Rx[2, 2]) },
            {   (Rzy[1, 0] * Rx[0, 0]) + (Rzy[1, 1] * Rx[1, 0]) + (Rzy[1, 2] * Rx[2, 0]),    (Rzy[1, 0] * Rx[0, 1]) + (Rzy[1, 1] * Rx[1, 1]) + (Rzy[1, 2] * Rx[2, 1]),    (Rzy[1, 0] * Rx[0, 2]) + (Rzy[1, 1] * Rx[1, 2]) + (Rzy[1, 2] * Rx[2, 2]) },
            {   (Rzy[2, 0] * Rx[0, 0]) + (Rzy[2, 1] * Rx[1, 0]) + (Rzy[2, 2] * Rx[2, 0]),    (Rzy[2, 0] * Rx[0, 1]) + (Rzy[2, 1] * Rx[1, 1]) + (Rzy[2, 2] * Rx[2, 1]),    (Rzy[2, 0] * Rx[0, 2]) + (Rzy[2, 1] * Rx[1, 2]) + (Rzy[2, 2] * Rx[2, 2]) }
        };

        Vector3 link_pos = linkpos[index];
        new_linkpos = new Vector3(
            (link_pos.x * R[0, 0]) + (link_pos.y * R[0, 1]) + (link_pos.z * R[0, 2]),
            (link_pos.x * R[1, 0]) + (link_pos.y * R[1, 1]) + (link_pos.z * R[1, 2]),
            (link_pos.x * R[2, 0]) + (link_pos.y * R[2, 1]) + (link_pos.z * R[2, 2])
            );
        index += 1;

        if (index != 6)
        {
            angles[index - 1] = 0;
            linkpos[index - 1] = new_linkpos;
            linkpos = update(index, linkpos);

            GetCoordinate(angles, linkpos, axes, index);
        }
        return linkpos;
        //List<Vector3> Rzy = new List<Vector3>();
        //Rzy.Add

    }

    private Vector3[] update(int index, Vector3[] linkpos)
    {
        if ((index + 1) < (linkpos.Length - 1))
        {
            linkpos[index + 1] = linkpos[index] + linkpos[index + 1];
            index += 1;
        }
        return linkpos;
    }


    void print(string result)
    {
        Vector3[] pos = new Vector3[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            pos[i] = joints[i].transform.position;
        }
        foreach(Vector3 val in pos)
        {
            UnityEngine.Debug.Log(val);
        }
        
    }

}
