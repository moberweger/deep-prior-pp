/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            initdepthsense.h                                              
 * author:          Abdi Dahir                           
 * modified:        May 9 2014                                               
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:                      
 *                                                                             
 * Imagebuffers defined here along with the depthsense start/stop ops
 * -----------------------------------------------------------------------------
 */

#include <stdint.h>

// map dimensions
static int32_t dW = 320;
static int32_t dH = 240;
static int32_t cW = 640;
static int32_t cH = 480;

static int dshmsz = dW*dH*sizeof(int16_t);
static int cshmsz = cW*cH*sizeof(uint8_t);
static int vshmsz = dW*dH*sizeof(int16_t);
static int ushmsz = dW*dH*sizeof(float);
static int hshmsz = dW*dH*sizeof(uint8_t);

// shared mem depth maps
extern int16_t *depthMap;
extern int16_t *depthFullMap;

// shared mem vertex maps
extern int16_t *vertexMap;
extern int16_t *vertexFullMap;

extern float *vertexFMap;
extern float *vertexFFullMap;

// shared mem colour maps
extern uint8_t *colourMap;
extern uint8_t *colourFullMap;

// shared mem accel maps
extern float *accelMap;
extern float *accelFullMap;

// shared mem uv maps
extern float *uvMap;
extern float *uvFullMap;

// frame counters
extern uint32_t g_aFrames;
extern uint32_t g_cFrames;
extern uint32_t g_dFrames;

// intrinsics
extern float g_dIntrinsics[9];
extern float g_cIntrinsics[9];

// extrinsics
extern float g_Extrinsics[12];

extern "C" {
    void killds();
    void initds();
}
