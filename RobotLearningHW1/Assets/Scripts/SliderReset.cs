using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class SliderReset : MonoBehaviour, IEndDragHandler
{

    private Slider me;

    void Awake()
    {
        me = gameObject.GetComponent<Slider>();
    }


    public void OnEndDrag(PointerEventData data)
    {
        me.value = 0f;
        
    }

    void LateUpdate()
    {
        // call this after all necessary processing updates are done
        if(Input.GetMouseButtonUp(0))
        {
            me.value = 0f;
        }

    }
}
