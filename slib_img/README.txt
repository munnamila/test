import slib_img as si

I = si.Img(path)

I.size()
#print the size of picture

I.gray()
#make picture become gray picture

I.flip
#右左回転

I.choose_face_dlib(dim)
#choose the area of face
#アフィン変換:YES
#default dim is 128x128

I.choose_face_cv()
#choose the area of face
#アフィン変換:NO
#size:64x64

I.show()
#show the picture by plt

I.save()
#save the picture to .png

I.autoencoder()
#return encoded and decoded

I.cut_pic(l_or_r)
#choose the left or right part of the picture

I.resize(height, weight)
#resize the picture

si.show_img()
