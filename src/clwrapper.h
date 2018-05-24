#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <exception>

class CLWrapper
{
private:
  cl_context context;
  cl_context_properties *properties;
  cl_kernel kernel;
  cl_command_queue command_queue;
  cl_program program;
  cl_int cl_err;

  cl_uint num_of_platforms;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_uint num_of_devices;
  cl_int arg_counter;

  std::vector<cl_mem> mem_objects;
public:
  CLWrapper()
    : num_of_platforms (0), num_of_devices (0), arg_counter (0)
  {
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
      {
	fprintf(stderr, "Unable to get platform id\n");
	exit(1);
      }
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id,
		       &num_of_devices) != CL_SUCCESS)
      {
	fprintf(stderr, "Unable to get device id\n");
	exit(1);
      }

    properties = new cl_context_properties[3];

    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties) platform_id;
    properties[2] = 0;

    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &cl_err);
    command_queue = clCreateCommandQueue(context, device_id, 0, &cl_err);
  }
  ~CLWrapper()
  {
    for (auto& obj : mem_objects)
      {
	clReleaseMemObject(obj);
      }

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    delete [] properties;
  }

  static void check_code(cl_int code, std::string msg)
  {
    if (code != CL_SUCCESS)
      {
	throw std::runtime_error(msg);
      }
  }


  cl_kernel build(std::string src_path, std::string kernel_name)
  {
    FILE *fp = fopen("update_block.cl", "r");
    fseek(fp, 0L, SEEK_END);
    size_t filelen = ftell(fp);
    rewind(fp);

    char *kernel_src = (char*) malloc(sizeof(char) * (filelen + 1));
    size_t readlen = fread(kernel_src, 1, filelen, fp);
    if (readlen != filelen)
      {
	fprintf(stderr, "Error reading cl code\n");
	exit(1);
      }
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src,
					NULL, &cl_err);
    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
      {
	printf("Error building program\n");
	char buffer[4096];
	// get the build log
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			      sizeof(buffer), buffer, NULL);
	printf("--- Build Log -- \n %s\n", buffer);
	exit(1);
      }
    kernel = clCreateKernel(program, kernel_name.c_str(), &cl_err);
    if (cl_err != CL_SUCCESS)
      {
	fprintf(stderr, "Error creating kernel\n");
	exit(1);
      }

    return kernel;
  }

  void run(size_t global)
  {
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    arg_counter = 0;
  }

  template <typename T>
  void in(std::vector<T>& data)
  {
    cl_mem object;
    object = clCreateBuffer(context, CL_MEM_READ_ONLY ,
			    data.size() * sizeof(T), NULL, &cl_err);
    mem_objects.push_back(object);
    check_code(cl_err, "Create buffer failed");
    cl_err = clEnqueueWriteBuffer(command_queue, object, CL_FALSE, 0,
				   sizeof(T) * data.size(), data.data(), 0
				   , NULL, NULL);
    check_code(cl_err, "Write buffer failed");
    cl_err = clSetKernelArg(kernel, arg_counter, sizeof(cl_mem),
			     &mem_objects[mem_objects.size()-1]);
    check_code(cl_err, "Set kernel arg:" + std::to_string(arg_counter) +
	       " failed");
    arg_counter ++;
  }

  template <typename T>
  void in(T& data, size_t size)
  {
    cl_mem object;
    object = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(T),
			    NULL, &cl_err);
    mem_objects.push_back(object);
    check_code(cl_err, "Creating buffer for array failed.");

    cl_err = clEnqueueWriteBuffer(command_queue, object, CL_FALSE, 0,
				  sizeof(T) * size, (void*)data, 0,
				  NULL, NULL);
    check_code(cl_err, "Writing buffer for array failed");
    cl_err = clSetKernelArg(kernel, arg_counter, sizeof(cl_mem),
			    &mem_objects[mem_objects.size()-1]);
    arg_counter ++;
  }
  template <typename T>
  void in(T scalar)
  {
    cl_err = clSetKernelArg(kernel, arg_counter, sizeof(T), &scalar);
    check_code(cl_err, "Set scalar arg:" + std::to_string(arg_counter) +
	       " failed");
    arg_counter ++;
  }

  cl_context get_context() const
  {
    return context;
  }
  cl_kernel get_kernel() const
  {
    return kernel;
  }
  cl_command_queue get_command_queue() const
  {
    return command_queue;
  }

  cl_int increase_counter (cl_int i=1)
  {
    arg_counter += i;
    return arg_counter;
  }
};
