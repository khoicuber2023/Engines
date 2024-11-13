import glfw
from OpenGL.GL import *
import glm
import numpy as np
from PIL import Image
from opensimplex import OpenSimplex
import random

vaos = []
vbos = []
ebos = []
positions = []
normals = []
textures = []

vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec3 a_normal;

out vec3 fragColor;

out vec2 outTexCoord;
out vec3 v_normal;
out vec3 v_fragPos;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    outTexCoord = texCoord;
    v_fragPos = vec3(model * vec4(position, 1.0));
    v_normal = mat3(transpose(inverse(model))) * a_normal; // Transform normal to world space
}
"""

fragment_shader_source = """
#version 330 core
in vec2 outTexCoord;

in vec3 v_normal;
in vec3 v_fragPos;

uniform vec3 lightPos; // Light position
uniform vec3 viewPos;  // Camera position
uniform vec3 lightColor; // Light color
uniform vec3 objectColor; // Object color (solid color of the object)
uniform sampler2D texture1; 

out vec4 finalColor;

void main()
{
    // Ambient lighting
    float ambientStrength = 1.0;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(lightPos - v_fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular lighting (Phong model)
    vec3 viewDir = normalize(viewPos - v_fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  // Reflect light direction around normal

    float specStrength = 1.6; // Specular strength (adjustable)
    float shininess = 32.0;   // Shininess factor (adjustable)

    // Phong specular highlight calculation
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specStrength * spec * lightColor;

    // Combine results
    vec3 result = (ambient + diffuse + specular) * objectColor * texture(texture1,outTexCoord).rgb * 0.85; // Apply the lighting to the object color

    // Output final color
    finalColor = vec4(result, 1.0);
}

"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(f"ERROR::SHADER::COMPILATION_FAILED\n{info_log.decode()}")
        glDeleteShader(shader)  # Delete if compilation failed
        return None  # Return None to indicate failure
    return shader  # Return shader only if compiled successfully

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)


    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

class Camera:
    def __init__(self, position=(0, 0, 3), up=(0, 1, 0), yaw=90.0, pitch=0.0, fov=65, aspect_ratio=1.0):
        self.position = glm.vec3(position)
        self.up = glm.vec3(up)
        self.front = glm.vec3(0.0, 0.0, 1.0)
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.speed = 4.0
        self.sensitivity = 0.1
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)
    
    def get_projection_matrix(self):
        return glm.perspective(glm.radians(self.fov), self.aspect_ratio, 0.01, 10000.0)

    def process_keyboard(self, direction, delta_time):
        velocity = self.speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "UP":
            self.position += self.up * velocity
        if direction == "DOWN":
            self.position -= self.up * velocity
        if direction == "LEFT":
            self.position -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if direction == "RIGHT":
            self.position += glm.normalize(glm.cross(self.front, self.up)) * velocity

    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos  # Y is inverted
        self.last_x = xpos
        self.last_y = ypos

        xoffset *= self.sensitivity
        yoffset *= self.sensitivity

        self.yaw += xoffset
        self.pitch += yoffset
        self.pitch = max(-89.0, min(self.pitch, 89.0))

        front = glm.vec3()
        front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        front.y = glm.sin(glm.radians(self.pitch))
        front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)

class Raycasting:
    def __init__(self, camera : Camera):
        self.camera = camera

    def cast_ray(self, distance):
        int_cam_pos = glm.vec3(int(self.camera.position.x),int(self.camera.position.y),int(self.camera.position.z))
        int_front_pos = glm.vec3(int(self.camera.front.x),int(self.camera.front.y),int(self.camera.front.z)) * distance
        return int_cam_pos + int_front_pos

class Texture:
    def __init__(self):...

    def load_texture(self, file_path):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)  # Sử dụng mipmap
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        try:
            img = Image.open(file_path).transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.convert("RGBA").tobytes()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)  # Tạo mipmap
        except Exception as e:
            print(f"Error loading texture: {e}")
            return None
        return texture_id

class CubeWithTexture:
    def __init__(self, text_id, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), face : list = None):
        self.position = glm.vec3(position)
        self.rotation = glm.vec3(rotation)
        self.scale = glm.vec3(scale)
        self.model_matrix = glm.mat4(1.0)
        self.texture_id = text_id
        self.face = face
        self.indices1 = []
    
        self.vertices = np.array([
            # Positions         # Texture Coords  # Normals
            # Front face
            -0.5, -0.5, 0.5,    0.0, 0.0,      0.0, 0.0, 1.0,
             0.5, -0.5, 0.5,    1.0, 0.0,      0.0, 0.0, 1.0,
             0.5,  0.5, 0.5,    1.0, 1.0,      0.0, 0.0, 1.0,
            -0.5,  0.5, 0.5,    0.0, 1.0,      0.0, 0.0, 1.0,
            # Back face
            -0.5, -0.5, -0.5,  0.0, 0.0,       0.0, 0.0, -1.0,
             0.5, -0.5, -0.5,  1.0, 0.0,       0.0, 0.0, -1.0,
             0.5,  0.5, -0.5,  1.0, 1.0,       0.0, 0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0, 1.0,       0.0, 0.0, -1.0,
            # Left face
            -0.5, -0.5, -0.5,  0.0, 0.0,       -1.0, 0.0, 0.0,
            -0.5, -0.5, 0.5,   1.0, 0.0,       -1.0, 0.0, 0.0,
            -0.5,  0.5, 0.5,   1.0, 1.0,       -1.0, 0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 1.0,       -1.0, 0.0, 0.0,
            # Right face
            0.5, -0.5, -0.5,   0.0, 0.0,        1.0, 0.0, 0.0,
            0.5, -0.5, 0.5,    1.0, 0.0,        1.0, 0.0, 0.0,
            0.5,  0.5, 0.5,    1.0, 1.0,        1.0, 0.0, 0.0,
            0.5,  0.5, -0.5,   0.0, 1.0,        1.0, 0.0, 0.0,
            # Bottom face
            -0.5, -0.5, -0.5,  0.0, 0.0,        0.0, -1.0, 0.0,
             0.5, -0.5, -0.5,  1.0, 0.0,        0.0, -1.0, 0.0,
             0.5, -0.5, 0.5,   1.0, 1.0,        0.0, -1.0, 0.0,
            -0.5, -0.5, 0.5,   0.0, 1.0,        0.0, -1.0, 0.0,
            # Top face
            -0.5,  0.5, -0.5,  0.0, 0.0,        0.0, 1.0, 0.0,
             0.5,  0.5, -0.5,  1.0, 0.0,        0.0, 1.0, 0.0,
             0.5,  0.5, 0.5,   1.0, 1.0,        0.0, 1.0, 0.0,
            -0.5,  0.5, 0.5,   0.0, 1.0,        0.0, 1.0, 0.0
        ], dtype='float32')

        self.render_face()

        self.indices = np.array(self.indices1, dtype='uint32')

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        vaos.append(self.vao)
        vbos.append(self.vbo)
        ebos.append(self.ebo)

        self.setup_buffers()

    def render_face(self):
        if self.face[0] is not None:
            self.indices1.extend([0,  1,  2,  2,  3,  0])
        if self.face[1] is not None:
            self.indices1.extend([4,  5,  6,  6,  7,  4])
        if self.face[2] is not None:
            self.indices1.extend([8,  9, 10, 10, 11,  8])
        if self.face[3] is not None:
            self.indices1.extend([12, 13, 14, 14, 15, 12])
        if self.face[4] is not None:
            self.indices1.extend([16, 17, 18, 18, 19, 16])
        if self.face[5] is not None:
            self.indices1.extend([20, 21, 22, 22, 23, 20])
        
    def setup_buffers(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(20))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_model_matrix(self):
        self.model_matrix = glm.translate(glm.mat4(1.0), self.position)
        self.model_matrix = glm.rotate(self.model_matrix, glm.radians(self.rotation.x), glm.vec3(1, 0, 0))
        self.model_matrix = glm.rotate(self.model_matrix, glm.radians(self.rotation.y), glm.vec3(0, 1, 0))
        self.model_matrix = glm.rotate(self.model_matrix, glm.radians(self.rotation.z), glm.vec3(0, 0, 1))
        self.model_matrix = glm.scale(self.model_matrix, self.scale)

    def draw(self, shader_program, model_location):
        self.update_model_matrix()
        glUseProgram(shader_program)
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(self.model_matrix))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

class App:
    def __init__(self):
        self.width,self.height = 1280,720
        self.OpenGL_Version = (3,3)
        if not glfw.init():
            raise Exception("GLFW could not be initialized")

        self.window = glfw.create_window(self.width , self.height, "KEngine | OpenGL", None, None)
        glfw.set_window_pos(self.window, 40, 35)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window could not be created")
        
        glfw.make_context_current(self.window)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.OpenGL_Version[0])  # Set OpenGL major version
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.OpenGL_Version[1])  # Set OpenGL minor version
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # Core profile is common

        self.texture = Texture()
        self.texture1 = self.texture.load_texture('textures/grass_top.png')
        self.texture2 = self.texture.load_texture('textures/Tree.jpg')
        self.texture3 = self.texture.load_texture('textures/dirt.jpg')
        self.texture4 = self.texture.load_texture('textures/bedrock.png')

        self.shape_item = []

        self.chunk = Chunk(self,chunk_size=(16,16))
        self.chunk.generate_chunk()

        normalnew1 = self.replace_duplicates_with_none(normals)
        normalnew2 = [tuple(normalnew1[i:i+6]) for i in range(0, len(normalnew1), 6)]
        
        for pos in positions:
            id = positions.index(pos)
            cube = CubeWithTexture(text_id = textures[id],position = pos,face = normalnew2[id])
            self.shape_item.append(cube)

        positions.clear()
        textures.clear()
        normals.clear()
        normalnew1.clear()
        normalnew2.clear()

        glViewport(0, 0, self.width,self.height)

        self.delta_time = 0
        self.fps = 0

        self.camera = Camera(fov = 70, aspect_ratio = self.width/self.height)
        
        glfw.set_cursor_pos_callback(self.window, self.camera.mouse_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        self.last_frame_time = glfw.get_time()

        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

        glUseProgram(self.shader_program)
        
        self.model_location = glGetUniformLocation(self.shader_program, "model")
        self.view_location = glGetUniformLocation(self.shader_program, "view")
        self.projection_location = glGetUniformLocation(self.shader_program, "projection")
        self.light_pos_loc = glGetUniformLocation(self.shader_program, "lightPos")
        self.view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        self.light_color_loc = glGetUniformLocation(self.shader_program, "lightColor")
        self.object_color_loc = glGetUniformLocation(self.shader_program, "objectColor")

        light_pos = np.array([30, 60, 30], dtype=np.float32)
        light_color = np.array([0.85,0.85,0.85], dtype=np.float32)
        object_color = np.array([1, 1, 1], dtype=np.float32)

        glUniform3fv(self.light_pos_loc, 1, light_pos)
        glUniform3fv(self.light_color_loc, 1, light_color)
        glUniform3fv(self.object_color_loc, 1, object_color)

    def get_texture_by_id(self,text_id):
        if text_id == 1:
            return self.texture1
        if text_id == 2:
            return self.texture2
        if text_id == 3:
            return self.texture3
        if text_id == 4:
            return self.texture4
        
    def window_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)

    def replace_duplicates_with_none(self, data):
        seen = {}  # Dùng từ điển để đếm số lần xuất hiện của mỗi phần tử
        result = []
        for item in data:
            if item in seen:
                seen[item] += 1
                if seen[item] == 2:  # Chỉ thay thế sau khi gặp lần thứ 2
                    result.append(None)
                    result[result.index(item)] = None  # Thay phần tử trước đó cũng bằng None
            else:
                seen[item] = 1
                result.append(item)
        
        return result
    
    def add_normals(self, pos: glm.vec3):
        front  = pos + glm.vec3(0, 0, 0.5)
        back   = pos - glm.vec3(0, 0, 0.5)
        right  = pos + glm.vec3(0.5, 0, 0)
        left   = pos - glm.vec3(0.5, 0, 0)
        top    = pos + glm.vec3(0, 0.5, 0)
        bottom = pos - glm.vec3(0, 0.5, 0)
        normals.extend([front, back, left, right, bottom, top])
    
    def is_point_in_view(self, point):
        # Lấy ma trận chiếu và ma trận nhìn của camera
        view_matrix = self.camera.get_view_matrix()
        projection_matrix = self.camera.get_projection_matrix()

        # Biến đổi điểm từ không gian thế giới sang không gian camera (view space)
        point_view_space = glm.vec4(point[0], point[1], point[2], 1.0)
        point_in_camera_space = view_matrix * point_view_space

        # Biến đổi điểm từ không gian camera (view space) sang không gian clip (clip space)
        point_in_clip_space = projection_matrix * point_in_camera_space

        # Kiểm tra xem w có bằng 0 không trước khi chuyển sang NDC
        if point_in_clip_space.w == 0.0:
            return False  # Nếu w = 0, điểm nằm ngoài phạm vi hợp lệ

        # Chuyển đổi từ không gian clip sang không gian NDC (Normalized Device Coordinates)
        # NDC có giá trị từ -1 đến 1 cho x, y và z
        x_ndc = point_in_clip_space.x / point_in_clip_space.w
        y_ndc = point_in_clip_space.y / point_in_clip_space.w
        z_ndc = point_in_clip_space.z / point_in_clip_space.w

        # Kiểm tra nếu tọa độ x, y, z trong phạm vi [-1, 1]
        if -10 <= x_ndc <= 10 and -10 <= y_ndc <= 10 and -10 <= z_ndc <= 10:
            return True  # Điểm nằm trong tầm nhìn
        return False  # Điểm không nằm trong tầm nhìn
    
    def run(self):
        glEnable(GL_DEPTH_TEST)

        while not glfw.window_should_close(self.window):
            self.update()
            self.handle_event()
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.clean_up()
        glfw.terminate()

    def render(self):
        for item in self.shape_item:
            item.draw(self.shader_program, self.model_location)
    
    def update(self):
        glfw.set_window_size_callback(self.window, self.window_size_callback)
        current_frame_time = glfw.get_time()
        self.delta_time = current_frame_time - self.last_frame_time
        self.last_frame_time = current_frame_time

        if self.delta_time != 0:
            self.fps = 1 / self.delta_time
            glfw.set_window_title(self.window,f'KEngine | OpenGL {self.OpenGL_Version[0]}.{self.OpenGL_Version[1]} | FPS: {self.fps :.2f} | Number of shape: {len(self.shape_item)}')

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1,0.16,0.35,1)
            
        view_pos = np.array([self.camera.position.x,self.camera.position.y,self.camera.position.z], dtype=np.float32)
        glUniform3fv(self.view_pos_loc, 1, view_pos)

        view_matrix = self.camera.get_view_matrix()
        projection_matrix = self.camera.get_projection_matrix()

        glUniformMatrix4fv(self.view_location, 1, GL_FALSE, glm.value_ptr(view_matrix))
        glUniformMatrix4fv(self.projection_location, 1, GL_FALSE, glm.value_ptr(projection_matrix))

    def handle_event(self):
        if glfw.get_key(self.window, glfw.KEY_F1) == glfw.PRESS:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        if glfw.get_key(self.window, glfw.KEY_F2) == glfw.PRESS:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.camera.process_keyboard("UP", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.camera.process_keyboard("DOWN", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.process_keyboard("FORWARD", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.process_keyboard("BACKWARD", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.process_keyboard("LEFT", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.process_keyboard("RIGHT", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            self.clean_up()
            quit()

    def clean_up(self):
        glDeleteVertexArrays(len(vaos), vaos)
        glDeleteBuffers(len(vbos), vbos)
        glDeleteBuffers(len(ebos), ebos)

class Chunk:
    def __init__(self, app, chunk_size):
        self.app = app
        self.chunk_size = chunk_size
        self.width, self.depth = self.chunk_size[0], self.chunk_size[1]
        self.cave_depth = 5
        self.noise = OpenSimplex(seed=1)
        self.positions = []

    def generate_chunk(self):
        for x in range(self.width):
            for z in range(self.depth):
                noise = self.noise.noise2(x * 0.08, z * 0.08)
                y = int(noise * 6) + 1  # Adjusting noise height
                positions.append((x, y, z))  # Store the generated position
                self.positions.append((x, y, z))  # Store the generated position
                textures.append(1)  # Default block texture (could be dirt, stone, etc.)
                self.app.add_normals(glm.vec3(x, y, z))  # Add the normal to your app
                self.generate_cave((x, y, z))

        max_y_pos = max(self.positions, key=lambda x: x[1])
        min_y_pos = min(self.positions, key=lambda x: x[1])
        self.generate_tree(max_y_pos)
        self.generate_tree(min_y_pos)
        self.positions.clear()

    def generate_tree(self,pos):
        x , y , z = pos
        tree_height = random.randint(3,4)
        for i in range(1,tree_height):
            positions.append((x , y + i , z))
            textures.append(2)  # Default block texture (could be dirt, stone, etc.)
            self.app.add_normals(glm.vec3(x , y + i , z))  # Add the normal to your app
        self.generate_tree_leaves(x,tree_height + y,z)
    
    def generate_tree_leaves(self, x, y, z):
        leaf_positions = [
            (x, y + 1, z), 
            (x + 1, y, z), (x - 1, y, z),
            (x, y, z + 1), (x, y, z - 1),
            (x + 1, y, z + 1), (x - 1, y, z - 1),
            (x + 1, y, z - 1), (x - 1, y, z + 1),
            (x + 1, y - 1, z), (x - 1, y - 1, z),
            (x, y - 1, z + 1), (x, y - 1, z - 1),
            (x + 1, y - 1, z + 1), (x - 1, y - 1, z - 1),
            (x + 1, y - 1, z - 1), (x - 1, y - 1, z + 1),  # Additional leaf spread            
        ]
        for (lx, ly, lz) in leaf_positions:
            positions.append((lx, ly + 1, lz))  # Add leaf block positions
            textures.append(3)  # Texture for leaves
            self.app.add_normals(glm.vec3(lx, ly + 1, lz))  # Add normal for leaves

    def generate_cave(self, pos):
        x, y, z = pos  # Unpack position
        for height in range(-self.cave_depth, y):
            if glm.simplex(glm.vec3(x * 0.08,height * 0.08,z * 0.08)) * 6 <= 0.2:
                position = (x, height, z)
                positions.append(position)
                textures.append(random.randint(3,4))
                self.app.add_normals(glm.vec3(x, height, z))

if __name__ == "__main__":
    app = App()
    app.run()
