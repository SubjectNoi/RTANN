#pragma once
#ifndef JUNO_RT_H_
#define JUNO_RT_H_

#include "utils.hpp"

struct Params {
    OptixTraversableHandle handle;
    int magic;
};

struct RayGenData {
    // float3 cam_eye;
    // float3 camera_u, camera_v, camera_w;
    float3* ray_origin;
    // unsigned int* ray_hit;
};

struct MissData {
    float r, g, b;
};

struct HitGroupData {
    unsigned int* hit_record;
    unsigned int* ray_info;
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}


namespace juno {

template <typename T>
class juno_rt {
private:
    OptixDeviceContext              context             = nullptr;
    OptixDeviceProperty             property            = {};
    CUcontext                       cuCtx               = 0;
    OptixDeviceContextOptions       options             = {};

    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer;
    CUdeviceptr                     d_params;
    OptixAccelBuildOptions          accel_options       = {};

    OptixPipeline                   pipeline            = nullptr;
    OptixShaderBindingTable         sbt                 = {};
    char                            log[2048];

    float3*                         d_ray_origin;
    float3*                         d_ray_origin_whole;
    Params                          params;
    unsigned int*                   d_hit_record;
    int                             hitable_num;
    int                             dim_pair;

    float factors[64] = {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431, 274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819, 244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
    // std::vector<int> nl16 = {2, 7, 8, 10, 28, 30, 32, 33, 51, 52, 63, 68, 71, 81, 88, 98, 102, 109, 124, 132, 148, 150, 155, 158, 165, 169, 181, 184, 190, 193, 197, 207, 211, 214, 218, 225, 226, 230, 249, 254, 264, 272, 284, 320, 321, 323, 324, 357, 359, 386, 413, 418, 423, 432, 434, 443, 444, 446, 452, 454, 457, 461, 469, 472, 475, 476, 480, 484, 485, 486, 490, 497, 501, 515, 524, 536, 537, 539, 552, 558, 576, 580, 592, 599, 600, 602, 614, 617, 627, 640, 642, 650, 651, 653, 657, 661, 666, 667, 679, 681, 708, 709, 713, 727, 741, 769, 779, 799, 800, 803, 812, 821, 837, 841, 861, 881, 898, 899, 900, 901, 902, 914, 916, 930, 947, 948, 966, 995, 1011, 1012, 1029, 1034, 1042, 1044, 1045, 1050, 1061, 1063, 1078, 1093, 1105, 1107, 1119, 1131, 1142, 1146, 1154, 1162, 1163, 1165, 1168, 1169, 1172, 1174, 1184, 1186, 1188, 1196, 1250, 1274, 1292, 1340, 1350, 1362, 1381, 1388, 1408, 1414, 1421, 1431, 1440, 1449, 1458, 1462, 1465, 1468, 1469, 1474, 1478, 1485, 1488, 1491, 1507, 1510, 1514, 1520, 1530, 1533, 1537, 1539, 1543, 1544, 1548, 1552, 1555, 1556, 1557, 1558, 1569, 1571, 1574, 1594, 1598, 1600, 1602, 1607, 1611, 1614, 1616, 1617, 1620, 1623, 1626, 1627, 1631, 1633, 1634, 1646, 1653, 1663, 1664, 1668, 1669, 1676, 1677, 1683, 1692, 1693, 1694, 1696, 1697, 1701, 1702, 1705, 1708, 1709, 1710, 1711, 1716, 1725, 1729, 1742, 1746, 1748, 1751, 1756, 1758, 1761, 1763, 1764, 1767, 1768, 1770, 1773, 1776, 1778, 1779, 1781, 1782, 1783, 1785, 1792, 1794, 1796, 1805, 1856, 1873, 1966, 1970, 1978, 2010, 2015, 2027, 2032, 2078, 2079, 2080, 2083, 2110, 2135, 2148, 2150, 2206, 2211, 2215, 2241, 2243, 2248, 2263, 2275, 2277, 2287, 2304, 2308, 2318, 2324, 2343, 2352, 2357, 2361, 2367, 2370, 2371, 2381, 2390, 2392, 2396, 2397, 2400, 2403, 2422, 2423, 2425, 2430, 2431, 2432, 2448, 2449, 2451, 2458, 2460, 2461, 2464, 2465, 2470, 2471, 2487, 2488, 2489, 2493, 2500, 2504, 2516, 2525, 2526, 2527, 2534, 2537, 2561, 2562, 2569, 2573, 2584, 2586, 2596, 2601, 2608, 2612, 2617, 2618, 2619, 2620, 2621, 2624, 2626, 2632, 2634, 2636, 2640, 2642, 2644, 2645, 2649, 2650, 2651, 2654, 2658, 2661, 2662, 2664, 2673, 2674, 2675, 2676, 2678, 2683, 2684, 2686, 2688, 2689, 2693, 2697, 2699, 2701, 2703, 2704, 2705, 2707, 2715, 2717, 2719, 2721, 2722, 2723, 2725, 2727, 2728, 2729, 2734, 2735, 2739, 2740, 2741, 2744, 2745, 2748, 2750, 2751, 2753, 2764, 2774, 2777, 2779, 2788, 2789, 2790, 2798, 2825, 2830, 2835, 2839, 2842, 2870, 2884, 2887, 2895, 2902, 2915, 2944, 2949, 2966, 2969, 2970, 2985, 2994, 2996, 2997, 2998, 3037, 3039, 3074, 3079, 3083, 3103, 3106, 3107, 3188, 3192, 3193, 3195, 3197, 3200, 3206, 3210, 3217, 3218, 3221, 3225, 3230, 3234, 3239, 3244, 3246, 3247, 3254, 3256, 3262, 3265, 3267, 3271, 3274, 3293, 3297, 3312, 3315, 3324, 3325, 3327, 3332, 3338, 3340, 3348, 3351, 3356, 3364, 3365, 3372, 3376, 3379, 3380, 3387, 3392, 3393, 3397, 3418, 3436, 3458, 3464, 3469, 3477, 3487, 3515, 3569, 3572, 3584, 3585, 3595, 3601, 3608, 3613, 3615, 3620, 3622, 3626, 3627, 3629, 3633, 3638, 3641, 3642, 3643, 3644, 3647, 3650, 3652, 3671, 3680, 3683, 3685, 3693, 3698, 3703, 3704, 3706, 3711, 3718, 3721, 3724, 3725, 3731, 3732, 3737, 3740, 3743, 3745, 3750, 3752, 3755, 3757, 3758, 3764, 3766, 3767, 3770, 3777, 3790, 3791, 3796, 3800, 3806, 3813, 3822, 3823, 3833, 3854, 3858, 3861, 3862, 3876, 3877, 3881, 3896, 3904, 3906, 3911, 3927, 3931, 3934, 3937, 3948, 3970, 3985, 3991, 4010, 4092, 4112, 4168, 4182, 4221, 4240, 4245, 4263, 4273, 4275, 4278, 4280, 4288, 4291, 4298, 4308, 4317, 4332, 4335, 4352, 4354, 4357, 4361, 4362, 4364, 4376, 4384, 4407, 4408, 4417, 4422, 4426, 4435, 4442, 4443, 4450, 4457, 4462, 4463, 4472, 4473, 4476, 4477, 4478, 4482, 4485, 4489, 4493, 4494, 4496, 4498, 4503, 4511, 4512, 4518, 4521, 4524, 4528, 4529, 4532, 4535, 4543, 4547, 4548, 4549, 4550, 4552, 4553, 4558, 4560, 4564, 4566, 4570, 4574, 4579, 4588, 4590, 4594, 4637, 4640, 4649, 4652, 4663, 4701, 4705, 4708, 4712, 4730, 4731, 4767, 4786, 4796, 4802, 4805, 4808, 4818, 4820, 4824, 4825, 4827, 4831, 4834, 4838, 4840, 4847, 4851, 4856, 4859, 4867, 4868, 4869, 4874, 4877, 4878, 4880, 4882, 4885, 4888, 4891, 4893, 4899, 4900, 4901, 4906, 4912, 4914, 4917, 4920, 4929, 4930, 4933, 4936, 4940, 4941, 4942, 4943, 4951, 4958, 4964, 4966, 4970, 4971, 4977, 4988, 4991, 4992, 4993, 5013, 5027, 5038, 5048, 5101, 5109, 5140, 5148, 5163, 5188, 5191, 5202, 5219, 5231, 5237, 5245, 5249, 5251, 5256, 5271, 5281, 5283, 5288, 5290, 5292, 5296, 5297, 5299, 5305, 5312, 5320, 5323, 5325, 5327, 5336, 5338, 5347, 5348, 5351, 5358, 5366, 5370, 5380, 5392, 5401, 5406, 5410, 5413, 5419, 5420, 5421, 5431, 5432, 5438, 5441, 5448, 5455, 5460, 5465, 5471, 5473, 5474, 5477, 5480, 5481, 5497, 5506, 5507, 5509, 5510, 5511, 5514, 5518, 5519, 5525, 5527, 5528, 5537, 5540, 5543, 5544, 5546, 5551, 5552, 5555, 5556, 5562, 5564, 5569, 5570, 5571, 5573, 5577, 5578, 5579, 5581, 5618, 5631, 5659, 5662, 5670, 5671, 5674, 5684, 5686, 5702, 5704, 5707, 5727, 5735, 5738, 5742, 5753, 5769, 5771, 5778, 5787, 5790, 5802, 5813, 5833, 5846, 5872, 5880, 5883, 5897, 5905, 5911, 5913, 5928, 5936, 5975, 5978, 5982, 5988, 6026, 6033, 6049, 6053, 6057, 6066, 6077, 6089, 6095, 6102, 6104, 6125, 6143, 6150, 6155, 6163, 6174, 6175, 6181, 6188, 6195, 6223, 6224, 6225, 6239, 6248, 6268, 6273, 6277, 6279, 6280, 6283, 6302, 6304, 6305, 6306, 6307, 6310, 6327, 6331, 6337, 6339, 6357, 6358, 6362, 6373, 6376, 6384, 6388, 6393, 6397, 6398, 6401, 6415, 6416, 6417, 6423, 6437, 6445, 6462, 6466, 6467, 6485, 6493, 6512, 6523, 6533, 6545, 6553, 6557, 6570, 6594, 6595, 6598, 6602, 6603, 6605, 6613, 6618, 6626, 6638, 6641, 6653, 6654, 6664, 6669, 6676, 6717, 6726, 6737, 6744, 6776, 6780, 6810, 6822, 6831, 6832, 6835, 6841, 6863, 6870, 6871, 6876, 6881, 6885, 6897, 6901, 6903, 6905, 6913, 6915, 6917, 6921, 6929, 6933, 6934, 6944, 6950, 6953, 6956, 6957, 6959, 6962, 6980, 6981, 6985, 6987, 6993, 6995, 6999, 7005, 7027, 7038, 7057, 7062, 7070, 7073, 7083, 7089, 7098, 7100, 7107, 7123, 7127, 7129, 7138, 7158, 7159, 7162, 7165, 7167, 7176, 7181, 7187, 7190, 7191, 7199, 7209, 7211, 7217, 7218, 7226, 7229, 7232, 7236, 7237, 7238, 7247, 7250, 7252, 7253, 7263, 7275, 7285, 7286, 7301, 7309, 7317, 7337, 7344, 7353, 7365, 7372, 7376, 7381, 7389, 7394, 7399, 7405, 7411, 7436, 7460, 7477, 7490, 7518, 7522, 7526, 7534, 7535, 7576, 7593, 7594, 7600, 7604, 7605, 7606, 7607, 7619, 7621, 7623, 7628, 7629, 7639, 7648, 7655, 7656, 7659, 7672, 7673, 7682, 7683, 7716, 7717, 7738, 7769, 7776, 7783, 7786, 7788, 7793, 7794, 7795, 7799, 7802, 7817, 7822, 7826, 7828, 7833, 7837, 7839, 7850, 7860, 7882, 7889, 7893, 7901, 7904, 7922, 7923, 7928, 7931, 7942, 7944, 7953, 7956, 7960, 7970, 7994, 8003, 8008, 8016, 8018, 8024, 8028, 8032, 8044, 8057, 8058, 8059, 8062, 8067, 8071, 8072, 8074, 8077, 8079, 8080, 8081, 8082, 8084, 8090, 8091, 8095, 8098, 8101, 8111, 8116, 8119, 8124, 8129, 8131, 8132, 8139, 8140, 8144, 8146, 8149, 8153, 8154, 8156, 8159, 8160, 8161, 8163, 8165, 8167, 8168, 8170, 8181, 8183, 8185, 8191, 8192, 8194, 8196, 8199, 8203, 8204, 8205, 8206, 8207, 8208, 8212, 8213, 8214, 8216, 8217, 8225, 8231, 8236, 8259, 8262, 8266, 8269, 8273, 8275, 8280, 8284, 8287, 8295, 8300, 8302, 8313, 8314, 8316, 8322, 8324, 8326, 8328, 8332, 8335, 8336, 8340, 8343, 8346, 8352, 8359, 8367, 8368, 8370, 8374, 8395, 8397, 8399, 8424, 8438, 8439, 8454, 8458, 8474, 8486, 8498, 8510, 8530, 8532, 8533, 8537, 8542, 8544, 8552, 8567, 8569, 8571, 8595, 8600, 8605, 8608, 8613, 8614, 8617, 8618, 8619, 8621, 8624, 8625, 8634, 8635, 8637, 8638, 8643, 8658, 8661, 8664, 8671, 8677, 8681, 8686, 8693, 8694, 8703, 8704, 8710, 8714, 8719, 8721, 8722, 8723, 8730, 8738, 8740, 8741, 8743, 8746, 8748, 8754, 8755, 8758, 8759, 8760, 8765, 8767, 8770, 8771, 8773, 8777, 8780, 8790, 8791, 8796, 8797, 8798, 8823, 8828, 8829, 8831, 8845, 8850, 8866, 8937, 9003, 9005, 9006, 9008, 9010, 9018, 9041, 9048, 9052, 9067, 9093, 9095, 9100, 9123, 9132, 9135, 9138, 9141, 9157, 9158, 9159, 9169, 9184, 9192, 9203, 9206, 9208, 9209, 9211, 9234, 9240, 9242, 9254, 9258, 9268, 9270, 9274, 9282, 9283, 9286, 9293, 9296, 9297, 9300, 9305, 9323, 9324, 9328, 9339, 9344, 9359, 9363, 9369, 9371, 9373, 9376, 9383, 9384, 9386, 9392, 9395, 9396, 9400, 9410, 9411, 9412, 9414, 9417, 9428, 9461, 9462, 9489, 9512, 9513, 9519, 9522, 9524, 9554, 9597, 9628, 9630, 9640, 9641, 9672, 9687, 9695, 9697, 9704, 9732, 9735, 9741, 9755, 9784, 9787, 9788, 9806, 9809, 9840, 9845, 9850, 9858, 9864, 9867, 9870, 9871, 9875, 9883, 9886, 9887, 9891, 9893, 9908, 9910, 9912, 9913, 9926, 9928, 9942, 9944, 9950, 9951, 9956, 9963, 9969, 9978, 9984, 9987, 9991, 9993, 9995, 9999};
    // std::vector<int> nl32 = {2, 8, 28, 32, 33, 42, 52, 56, 68, 78, 88, 89, 102, 111, 114, 115, 132, 139, 150, 154, 155, 158, 159, 166, 174, 184, 193, 202, 207, 253, 307, 311, 324, 359, 387, 393, 396, 426, 509, 552, 583, 641, 803, 812, 827, 832, 837, 842, 848, 861, 881, 896, 898, 899, 902, 913, 914, 930, 947, 959, 962, 972, 991, 995, 1021, 1040, 1044, 1066, 1068, 1073, 1111, 1142, 1188, 1362, 1403, 1408, 1444, 1458, 1470, 1485, 1491, 1548, 1568, 1571, 1574, 1655, 1798, 2403, 2410, 2422, 2430, 2448, 2461, 2462, 2471, 2487, 2489, 2494, 2497, 2498, 2517, 2525, 2539, 2544, 2547, 2548, 2561, 2569, 2573, 2574, 2584, 2586, 2601, 2609, 2611, 2616, 2618, 2619, 2624, 2630, 2632, 2634, 2639, 2640, 2642, 2651, 2654, 2658, 2661, 2662, 2664, 2673, 2674, 2676, 2678, 2681, 2683, 2684, 2689, 2697, 2699, 2701, 2704, 2705, 2706, 2707, 2710, 2711, 2728, 2729, 2733, 2734, 2739, 2740, 2744, 2745, 2751, 2754, 2756, 2758, 2759, 2772, 2776, 2779, 2784, 2787, 2789, 2790, 2795, 2798, 2810, 2835, 2994, 3458, 3487, 3569, 3800, 3822, 3852, 3854, 3862, 4191, 4417, 4435, 4443, 4447, 4451, 4456, 4464, 4466, 4473, 4484, 4485, 4543, 4549, 4552, 4560, 4564, 4566, 4570, 4571, 4579, 4590, 4594, 4836, 4945, 5350, 5355, 5438, 5510, 5630, 5686, 5717, 5742, 5778, 5790, 6004, 6017, 6066, 6089, 6095, 6298, 6314, 6433, 6833, 7027, 7032, 7054, 7070, 7083, 7089, 7098, 7107, 7113, 7129, 7162, 7165, 7190, 7204, 7208, 7215, 7217, 7224, 7236, 7237, 7247, 7250, 7253, 7258, 7263, 7269, 7275, 7285, 7286, 7300, 7301, 7315, 7327, 7376, 7411, 7414, 7477, 7607, 8016, 8023, 8044, 8080, 8101, 8149, 8201, 8215, 8501, 8510, 8567, 8579, 8605, 8617, 8625, 8648, 8664, 8736, 8744, 8755, 8757, 8780, 8797, 8828, 8829, 8831, 8924, 9062, 9138, 9212, 9234, 9240, 9265, 9280, 9282, 9301, 9311, 9388, 9697, 9736};
    
public:
    juno_rt(int _Q=10000, int _D=128) {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * 64 * 80));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin_whole), sizeof(float3) * _Q * (_D / 2) * NLISTS_MAX));
        cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params));
    }

    void constructCompleteBVHwithPQ(T**** _codebook_entry,
                                    int _coarse_grained_cluster_num,
                                    int _C, int _D, int _M,
                                    T** _stat,
                                    T _r,
                                    METRIC _metric,
                                    float alpha,
                                    float beta) 
    {
        struct timeval st, ed;
        float3* centers;
        float* radius;
        dim_pair = _D / _M;
        int prim_idx = 0;
        int num_sphere_per_dim_pair = _C;
        hitable_num = _coarse_grained_cluster_num * num_sphere_per_dim_pair * dim_pair;
        centers = new float3[hitable_num];
        radius = new float[hitable_num];
        for (int _c = 0; _c < _coarse_grained_cluster_num; _c++) {
            int c = _c;
        // for (int c = 0; c < _coarse_grained_cluster_num; c++) {
            for (int d = 0; d < dim_pair; d++) {
                // float _radius = _r * factors[d];
                float _radius = _r * 1.0;
                for (int n = 0; n < num_sphere_per_dim_pair; n++) {
                    float x = (1.0 * _codebook_entry[c][d][n][0]) / 100.0;
                    float y = (1.0 * _codebook_entry[c][d][n][1]) / 100.0;
                    // float factor = 1.0 * std::min(std::abs(x), std::abs(y));
                    float factor = alpha * std::min(std::abs(x), std::abs(y));
                    centers[_c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = make_float3(x, y, 1.0 * (c * 128 + 2 * d + 1));
                    // if (c == 432 && d == 0) printf("Prim %d:(%.6f, %.6f, %.6f)\n", prim_idx, x, y, 1.0 * (c * 128 + 2 * d + 1));
                    // radius[c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = static_cast<float>(0.45 + factor);
                    radius[_c * dim_pair * num_sphere_per_dim_pair + d * num_sphere_per_dim_pair + n] = static_cast<float>(beta+factor);
                    prim_idx++;
                }
            }
        }

        CUdeviceptr d_centers;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers), hitable_num * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centers), centers, hitable_num * sizeof(float3), cudaMemcpyHostToDevice));
    
        CUdeviceptr d_radius;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), hitable_num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), radius, hitable_num * sizeof(float), cudaMemcpyHostToDevice));

        uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers = &d_centers;
        sphere_input.sphereArray.numVertices = hitable_num;
        sphere_input.sphereArray.radiusBuffers = &d_radius;
        sphere_input.sphereArray.flags = sphere_input_flags;
        sphere_input.sphereArray.numSbtRecords = 1;

        gettimeofday(&st, NULL);
        constructBVHTreeWithPrimitives(&sphere_input, PRIMITIVE_TYPE_SPHERE);
        gettimeofday(&ed, NULL);
        elapsed("Constructing BVH", st, ed);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_centers)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_radius)));
    }

    // Construct a big whole BVH and hit with tag.
    // At most 256 clusters are supported, how to handle 1B dataset?
    void constructBVHwithPQ(int _label,             // BVH build for _label(th) cluster
                            T**** _codebook_entry,  // use _codebook_entry[_label][...][...][...]
                            int _C,                 // PQ_entry
                            int _D,                 // Dimension
                            int _M,                 // == 2
                            T** _stat,              // min(d), max(d)
                            T _r, 
                            METRIC _metric
                            ) 
    {
        float3* centers;
        float* radius;
        int num_circles_per_dimpair = _C;
        dim_pair = _D / _M;
        hitable_num = num_circles_per_dimpair * dim_pair;
        centers = new float3[hitable_num];
        radius = new float[hitable_num];
        for (int d = 0; d < dim_pair; d++) {
            float _radius = _r * factors[d];
            for (int n = 0; n < num_circles_per_dimpair; n++) {
                float x = 1.0 * _codebook_entry[_label][d][n][0];
                float y = 1.0 * _codebook_entry[_label][d][n][1];
                centers[d * num_circles_per_dimpair + n] = make_float3(x, y, 1.0 * (2 * d + 1));
                radius[d * num_circles_per_dimpair + n] = _radius;
            }
        }
        CUdeviceptr d_centers;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers), hitable_num * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centers), centers, hitable_num * sizeof(float3), cudaMemcpyHostToDevice));
    
        CUdeviceptr d_radius;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), hitable_num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), radius, hitable_num * sizeof(float), cudaMemcpyHostToDevice));

        uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers = &d_centers;
        sphere_input.sphereArray.numVertices = hitable_num;
        sphere_input.sphereArray.radiusBuffers = &d_radius;
        sphere_input.sphereArray.flags = sphere_input_flags;
        sphere_input.sphereArray.numSbtRecords = 1;

        constructBVHTreeWithPrimitives(&sphere_input, PRIMITIVE_TYPE_SPHERE);
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_centers)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_radius)));

    }

    void constructBVHforLabelWithRadius(int _label, T** _search_points, int* _search_points_labels, int _N, int _D, T** _stat, T _radius, METRIC _metric) {
        float3* vertices;
        int num_vertices;
        std::vector <int> point_index_of_label;
        for (int i = 0; i < _N; i++) {
            if (_search_points_labels[i] == _label) {
                point_index_of_label.push_back(i);
            }
        }
        int M;
        switch (_metric) {
            case METRIC_L2: {
                M = 2;
                dim_pair = _D / M;
                hitable_num = point_index_of_label.size(); 
                hitable_num = 64;
                num_vertices = hitable_num * TRIANGLE_PER_HITABLE * dim_pair;
                vertices = new float3[num_vertices];
                for (int d = 0; d < dim_pair; d++) {
                    // T a = _stat[d*2][1], b = _stat[d*2+1][1];
                    // T factor = sqrt(a * a + b * b);
                    // float factor = std::sqrt(_stat[d << 1][1] * _stat[d << 1][1] + _stat[d << 1 + 1][1] * _stat[d << 1 + 1][1]);
                    _radius = 0.25 * factors[d];
                    for (int n = 0; n < hitable_num; n++) {
                        T x = _search_points[point_index_of_label[n]][(d << 1)];
                        T y = _search_points[point_index_of_label[n]][(d << 1) + 1];
                        float3 v1 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v2 = make_float3(x - _radius, y + _radius, 1.0 * d + 1);
                        float3 v3 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v4 = make_float3(x - _radius, y - _radius, 1.0 * d + 1);
                        float3 v5 = make_float3(x + _radius, y + _radius, 1.0 * d + 1);
                        float3 v6 = make_float3(x + _radius, y - _radius, 1.0 * d + 1);
                        int base_idx = d * hitable_num * TRIANGLE_PER_HITABLE + n * TRIANGLE_PER_HITABLE;
                        vertices[base_idx + 0] = v1;
                        vertices[base_idx + 1] = v2;
                        vertices[base_idx + 2] = v3;
                        vertices[base_idx + 3] = v4;
                        vertices[base_idx + 4] = v5;
                        vertices[base_idx + 5] = v6;
                    }
                }
                CUdeviceptr d_vertices;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), num_vertices * sizeof(float3)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices, num_vertices * sizeof(float3), cudaMemcpyHostToDevice));

                const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
                OptixBuildInput triangle_input = {};
                triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangle_input.triangleArray.numVertices = static_cast<uint32_t>(num_vertices);
                triangle_input.triangleArray.vertexBuffers = &d_vertices;
                triangle_input.triangleArray.flags = triangle_input_flags;
                triangle_input.triangleArray.numSbtRecords = 1;
                
                constructBVHTreeWithPrimitives(&triangle_input, PRIMITIVE_TYPE_TRIANGLE);
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
                break;
            }
            case METRIC_MIPS:
                M = 3;
                break;
            case METRIC_COS:
                M = 3;
                break;
            default:
                M = -1;
                break;
        }

    }

    void constructBVHTreeWithPrimitives(OptixBuildInput* primitive_input, PRIMITIVE_TYPE prim_type) {
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, primitive_input, 1, &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, primitive_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                
        OptixModule module = nullptr;
        OptixModule triangle_module;
        OptixModuleCompileOptions module_compile_options = {};
        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 3;
        pipeline_compile_options.numAttributeValues = 3;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = (prim_type == PRIMITIVE_TYPE_SPHERE) ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
                
        std::string input;
        std::ifstream file(std::getenv("JUNO_ROOT") + std::string("/src/juno_rt.optixir"), std::ios::binary);
        if (file.good()) {
            std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
            input.assign(buffer.begin(), buffer.end());
        }

        size_t inputSize = input.size();
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options, input.c_str(), inputSize, log, &sizeof_log, &module));
        OptixBuiltinISOptions builtin_is_options = {};
        builtin_is_options.usesMotionBlur = false;

        builtin_is_options.builtinISModuleType = (prim_type == PRIMITIVE_TYPE_SPHERE) ? OPTIX_PRIMITIVE_TYPE_SPHERE : OPTIX_PRIMITIVE_TYPE_TRIANGLE;
        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &triangle_module));

        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;

        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &raygen_prog_group_desc, 
                                                1, 
                                                &program_group_options,
                                                log,
                                                &sizeof_log,
                                                &raygen_prog_group
                                                ));
                
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &miss_prog_group_desc, 
                                                1, 
                                                &program_group_options, 
                                                log, 
                                                &sizeof_log, 
                                                &miss_prog_group
                                                ));
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hitgroup_prog_group_desc.hitgroup.moduleIS = triangle_module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                                &hitgroup_prog_group_desc, 
                                                1, 
                                                &program_group_options, 
                                                log, 
                                                &sizeof_log, 
                                                &hitgroup_prog_group
                                                ));
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(context, 
                                            &pipeline_compile_options, 
                                            &pipeline_link_options, 
                                            program_groups, 
                                            sizeof(program_groups) / sizeof(program_groups[0]), 
                                            log, 
                                            &sizeof_log, 
                                            &pipeline
                                            ));
        OptixStackSizes stack_sizes = {};
        for (auto& prog_group: program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_satck_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 
                                                max_trace_depth, 
                                                0, 
                                                0, 
                                                &direct_callable_stack_size_from_traversal, 
                                                &direct_callable_stack_size_from_state, 
                                                &continuation_satck_size
                                                ));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 
                                              direct_callable_stack_size_from_traversal, 
                                                direct_callable_stack_size_from_state, 
                                                continuation_satck_size, 
                                                1
                                                ));

        // data of d_ray_origin should be feed runtime.
        CUdeviceptr raygen_record;
        size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        rg_sbt.data.ray_origin = d_ray_origin_whole;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));
        sbt.raygenRecord = raygen_record;

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        ms_sbt.data = {0.0f, 0.0f, 0.0f};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;  
                
        int d_hit_record_size = QUERY_BATCH_MAX * dim_pair * NLISTS_MAX * (MAX_ENTRY / 32);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(unsigned int) * d_hit_record_size));
        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        hg_sbt.data.hit_record = d_hit_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1; 

        params.handle = gas_handle;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));
    }

    void freeResources() {
        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        OPTIX_CHECK( optixDeviceContextDestroy( context ) );
    }

    void setRayOrigin(float3* ray_origin, int size) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ray_origin_whole), ray_origin, sizeof(float3) * size, cudaMemcpyHostToDevice));
    }

    void getRayHitRecord(unsigned int* hit_record, int size) {
        CUDA_CHECK(cudaMemcpy(hit_record, reinterpret_cast<void*>(d_hit_record), sizeof(unsigned int) * size * (MAX_ENTRY / 32), cudaMemcpyDeviceToHost));
    }

    // static void initRayOriginArray(int Q, int D, int M, int _nlists) {
    //     CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * Q * _nlists * D / M));
    // }

    OptixPipeline& getOptixPipeline() {
        return pipeline;
    }

    CUdeviceptr getDparams() {
        return d_params;
    }

    OptixShaderBindingTable* getSBT() {
        return &sbt;
    }

    unsigned int* getPrimitiveHit() {
        return d_hit_record;
    }

    OptixTraversableHandle& getGasHandle() {
        return gas_handle;
    }

    int getHitableNum() {
        return hitable_num * dim_pair;
    }
}; // class juno_rt

}; // namespace juno

#endif
