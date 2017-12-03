using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Rotate : MonoBehaviour {

    public Vector3 rotation_constraint;
    public float angle;
    public float anim_speed;
    public Slider slider;
	// Use this for initialization
	void Start () {
        //Debug.Log(transform.localRotation)
        previous_rotation = 0;
	}
	
	// Update is called once per frame
	void Update () {
        rotate_once(angle);
        rotate(slider.value);
    }

    void rotate(float value)
    {
        if(value != 0)
        {
            Quaternion to = Quaternion.Euler(Vector3.Scale(new Vector3(value, value, value), rotation_constraint));
            transform.localRotation = Quaternion.RotateTowards(transform.localRotation, transform.localRotation * to, anim_speed * Time.deltaTime);
        }
        
    }

    float previous_rotation;
    void rotate_once(float value)
    {
        if(value != 0)
        {
            Quaternion to = Quaternion.Euler(Vector3.Scale(new Vector3(value, value, value), rotation_constraint));
            transform.localRotation = Quaternion.RotateTowards(transform.localRotation, to, anim_speed * Time.deltaTime);
            previous_rotation = value;
            //Debug.Log("inside rotate_once");
        }
        
    }

    public void on_slider_moved()
    {
        //rotate(slider.value);
        //Debug.Log("I was called: " + slider.value);
    }

    public Vector3 GetRelativeAngle()
    {
        return transform.localEulerAngles;
    }
}
