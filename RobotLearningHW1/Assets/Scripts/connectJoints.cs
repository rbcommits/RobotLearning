using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class connectJoints : MonoBehaviour {

    // Use this for initialization
    void Start () {

	}
	
	// Update is called once per frame
	void Update () {
		
	}
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Vector3 point = this.transform.GetChild(0).position;
        foreach(Transform t in this.transform)
        {
            Gizmos.DrawLine(point, t.position);
            point = t.position;
        }

    }
}
