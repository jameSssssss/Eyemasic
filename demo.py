#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import _init_paths
import caffe
import cv2
import numpy as np
#from python_wrapper import *
import os
import sys
g_quiet=True
g_domasic=True
#import _init_paths
def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print ("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print "bb", boundingbox
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print '#################'
    #print 'boxes', boxes
    #print 'w,h', w, h
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    tmph=tmph.astype(int)
    tmpw=tmpw.astype(int)
    numbox = boxes.shape[0]

    #print 'tmph', tmph
    #print 'tmpw', tmpw

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0]
    y = boxes[:,1]
    ex = boxes[:,2]
    ey = boxes[:,3]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1).astype(int)
    dx = np.maximum(0, dx-1).astype(int)
    y = np.maximum(0, y-1).astype(int)
    x = np.maximum(0, x-1).astype(int)
    edy = np.maximum(0, edy-1).astype(int)
    edx = np.maximum(0, edx-1).astype(int)
    ey = np.maximum(0, ey-1).astype(int)
    ex = np.maximum(0, ex-1).astype(int)
    
    #print "dy"  ,dy 
    #print "dx"  ,dx 
    #print "y "  ,y 
    #print "x "  ,x 
    #print "edy" ,edy
    #print "edx" ,edx
    #print "ey"  ,ey 
    #print "ex"  ,ex 


    #print 'boxes', boxes
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print 'bboxA', bboxA
    #print 'w', w
    #print 'h', h
    #print 'l', l
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: [pick]
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print "1: x,y", x,y
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print "2: x,y", x,y
    else:
        score = map[x,y]
    '''
    #print "dx1.shape", dx1.shape
    #print 'map.shape', map.shape
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print '(x,y)',x,y
    #print 'score', score
    #print 'reg', reg

    return boundingbox_out.T



def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print (fmt % (time()-_tstart_stack.pop()))


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    #img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    

    #total_boxes = np.load('total_boxes.npy')
    #total_boxes = np.load('total_boxes_242.npy')
    #total_boxes = np.load('total_boxes_101.npy')

    
    # create scale pyramid
    scales = []
    #net1_start=cv2.getTickCount()
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print "scale:", scale


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        
        if boxes.shape[0] != 0:
            #print boxes[4:9]
            #print 'im_data', im_data[0:5, 0:5, 0], '\n'
            #print 'prob1', out['prob1'][0,0,0:3,0:3]

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    #print "[1]:",total_boxes.shape
    #print total_boxes
    #return total_boxes, [] 

    #net1_stop=cv2.getTickCount()
    #print "net1_time:",(net1_stop-net1_start)/cv2.getTickFrequency()
    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
       # print "[2]:",total_boxes.shape[0]
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        #print "[3]:",total_boxes.shape[0]
        #print regh
        #print regw
        #print 't1',t1
        #print total_boxes
        
        total_boxes = rerec(total_boxes) # convert box to square
        #print "[4]:",total_boxes.shape[0]
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        #print "[4.5]:",total_boxes.shape[0]
        #print total_boxes
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print total_boxes.shape
    #print total_boxes

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print 'tmph', tmph
        #print 'tmpw', tmpw
        #print "y,ey,x,ex", y, ey, x, ex, 
        #print "edy", edy

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((tmph[k], tmpw[k],3))
          
            #print "dx[k], edx[k]:", dx[k], edx[k]
            #print "dy[k], edy[k]:", dy[k], edy[k]
            #print "img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape
            #print "tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape

            tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
            #print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
            #print "tmp", tmp.shape
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print 'tempimg', tempimg[k,:,:,:].shape
            #print tempimg[k,0:5,0:5,0] 
            #print tempimg[k,0:5,0:5,1] 
            #print tempimg[k,0:5,0:5,2] 
            #print k
    
        #print tempimg.shape
        #print tempimg[0,0,0,:]
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print tempimg[0,:,0,0]
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print out['conv5-2'].shape
        #print out['prob1'].shape

        score = out['prob1'][:,1]
        #print 'score', score
        pass_t = np.where(score>threshold[1])[0]
        #print 'pass_t', pass_t
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        #print "[5]:",total_boxes.shape[0]
        #print total_boxes

        #print "1.5:",total_boxes.shape
        
        mv = out['conv5-2'][pass_t, :].T
        #print "mv", mv,mv.shape
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print 'pick', pick
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                #print "[6]:",total_boxes.shape[0]
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print "[7]:",total_boxes.shape[0]
                total_boxes = rerec(total_boxes)
                #print "[8]:",total_boxes.shape[0]
            
        #####
        # 2 #
        #####
        #print "2:",total_boxes.shape

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print 'tmpw', tmpw
            #print 'tmph', tmph
            #print 'y ', y
            #print 'ey', ey
            #print 'x ', x
            #print 'ex', ex
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((tmph[k], tmpw[k],3))
                tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            #print "[9]:",total_boxes.shape[0]
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                #print "[10]:",total_boxes.shape[0]
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print pick
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    #print "[11]:",total_boxes.shape[0]
                    points = points[pick, :]
                points=np.maximum(points,0)
            
                points[:,0:5]=np.minimum(points[:,0:5],img.shape[1])
                points[:,5:10]=np.minimum(points[:,5:10],img.shape[0])
            
                total_boxes=np.maximum(total_boxes,0)
                total_boxes[:,2]=np.minimum(total_boxes[:,2],img.shape[1])
                total_boxes[:,3]=np.minimum(total_boxes[:,3],img.shape[0])
    #####
    # 3 #
    #####
    #print "3:",total_boxes.shape
    
    return total_boxes, points


def track_face(img, minsize, RNet, ONet, threshold, fastresize, factor,face_boxes):
    

    
   
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    

    #total_boxes = np.load('total_boxes.npy')
    #total_boxes = np.load('total_boxes_242.npy')
    #total_boxes = np.load('total_boxes_101.npy')

    
    # create scale pyramid
    total_boxes=face_boxes.copy()
    numbox = total_boxes.shape[0]
    if numbox > 0:
        
        total_boxes = rerec(total_boxes) # convert box to square
        #print "[4]:",total_boxes.shape[0]
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        #print "[4.5]:",total_boxes.shape[0]
        #print total_boxes
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print total_boxes.shape
    #print total_boxes

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print 'tmph', tmph
        #print 'tmpw', tmpw
        #print "y,ey,x,ex", y, ey, x, ex, 
        #print "edy", edy

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((tmph[k], tmpw[k],3))
          
            #print "dx[k], edx[k]:", dx[k], edx[k]
            #print "dy[k], edy[k]:", dy[k], edy[k]
            #print "img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape
            #print "tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape

            tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
            #print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
            #print "tmp", tmp.shape
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print 'tempimg', tempimg[k,:,:,:].shape
            #print tempimg[k,0:5,0:5,0] 
            #print tempimg[k,0:5,0:5,1] 
            #print tempimg[k,0:5,0:5,2] 
            #print k
    
        #print tempimg.shape
        #print tempimg[0,0,0,:]
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print tempimg[0,:,0,0]
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print out['conv5-2'].shape
        #print out['prob1'].shape

        score = out['prob1'][:,1]
        #print 'score', score
        pass_t = np.where(score>threshold[1])[0]
        #print 'pass_t', pass_t
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        #print "[5]:",total_boxes.shape[0]
        #print total_boxes

        #print "1.5:",total_boxes.shape
        
        mv = out['conv5-2'][pass_t, :].T
        #print "mv", mv,mv.shape
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print 'pick', pick
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                #print "[6]:",total_boxes.shape[0]
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print "[7]:",total_boxes.shape[0]
                total_boxes = rerec(total_boxes)
                #print "[8]:",total_boxes.shape[0]
            
        #####
        # 2 #
        #####
        #print "2:",total_boxes.shape

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print 'tmpw', tmpw
            #print 'tmph', tmph
            #print 'y ', y
            #print 'ey', ey
            #print 'x ', x
            #print 'ex', ex
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((tmph[k], tmpw[k],3))
                tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            #print "[9]:",total_boxes.shape[0]
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                #print "[10]:",total_boxes.shape[0]
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print pick
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    #print "[11]:",total_boxes.shape[0]
                    points = points[pick, :]
            
                points=np.maximum(points,0)
            
                points[:,0:5]=np.minimum(points[:,0:5],img.shape[1])
                points[:,5:10]=np.minimum(points[:,5:10],img.shape[0])
            
                total_boxes=np.maximum(total_boxes,0)
                total_boxes[:,2]=np.minimum(total_boxes[:,2],img.shape[1])
                total_boxes[:,3]=np.minimum(total_boxes[:,3],img.shape[0])
            
    #####
    # 3 #
    #####
    #print "3:",total_boxes.shape

    return total_boxes, points

    
def initFaceDetector():
    minsize = 20
    caffe_model_path = "/home/duino/iactive/mtcnn/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    return (minsize, PNet, RNet, ONet, threshold, factor)
'''
def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]
    
    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    #tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    #toc()
    containFace = (True, False)[boundingboxes.shape[0]==0]
    return containFace, boundingboxes
'''
def createmasic(facial_points,bbw,bbh,old_masic,img_width,img_height,masic_type=1,width_ratio=6,height_ratio=30,threshold=0.8):
    '''
    马赛克位置生成，若与上一帧马赛克overlap则不变
    facial_points.shape=(face_num,10)  :face_num x [x1,x2,,,x5,y1,y2,,,y5]
    bbw,bbh :list of boundingbox width and height

    return:list[list[x1],list[y1],list[x2],list[y2]]
    '''

    masic_width = bbw / width_ratio
    masic_height = bbh / height_ratio
    if masic_type == 1:
        miny = np.minimum(facial_points[:, 5], facial_points[:, 6])
        maxy = np.maximum(facial_points[:, 5], facial_points[:, 6])

        newmasic = [facial_points[:, 0] - masic_width, miny - masic_height, facial_points[:, 1] + masic_width,
                        maxy + masic_height]
    newmasic = np.array(newmasic).astype(int)
    newmasic = newmasic.reshape((-1, 4))
    new_area = np.multiply(newmasic[:, 2] - newmasic[:, 0] + 1, newmasic[:, 3] - newmasic[:, 1] + 1)
    old_area = np.multiply(old_masic[:, 2] - old_masic[:, 0] + 1, old_masic[:, 3] - old_masic[:, 1] + 1)

    for i in range(newmasic.shape[0]):
        xx1 = np.maximum(newmasic[i, 0], old_masic[:, 0])
        yy1 = np.maximum(newmasic[i, 1], old_masic[:, 1])
        xx2 = np.minimum(newmasic[i, 2], old_masic[:, 2])
        yy2 = np.minimum(newmasic[i, 3], old_masic[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (new_area[i] + old_area - inter)
        pick = np.where(o > threshold)
        pick = np.array(pick)
        #print o
        #print pick.shape
        if (pick.shape[1] > 0):
            newmasic[i] = old_masic[pick[0, 0]]

    newmasic=np.fix(np.maximum(newmasic,0)).astype(int)
    newmasic[:,2]=np.minimum(newmasic[:,2],img_width)
    
    newmasic[:,3]=np.minimum(newmasic[:,3],img_height)

    
    return newmasic


def main():
    #imglistfile = "./file.txt"
    
    #imglistfile = "/home/duino/iactive/mtcnn/all.txt"
    #imglistfile = "./imglist.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file_n.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file.txt"
    tracknum=0
    
    orispeed=100
    trackspeed=orispeed
    track_hist=[]
    masic_type=1
    minsize = 20
    if len(sys.argv)!=3 :
        print("wrong input argument ")
        return 1
    videofile_s=sys.argv[1]
    writeVideoFile=sys.argv[2]
    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.8]
    threshold2=[0.3,0.4,0.4]
    factor = 0.709
    masic=np.array([[0,0,0,0]])
    caffe.set_mode_gpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    video=cv2.VideoCapture(videofile_s)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/2),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
    videoWriter = cv2.VideoWriter(writeVideoFile, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
    if  not videoWriter.isOpened():
        print("writer not open")
        return
    if not video.isOpened():
        print("capture not open")
        return 
    #error = []
    success, img = video.read()  
    while success:
        whole_begin=cv2.getTickCount()
        img=cv2.resize(img,(size[0],size[1]))
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
                  
        
        if tracknum==0:    
            del track_hist[:]
            tickfre=cv2.getTickFrequency()
            network_begin=cv2.getTickCount()
            boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
            network_end=cv2.getTickCount()
            if boundingboxes.shape[0]>0:
                masic=createmasic(points,boundingboxes[:,2]-boundingboxes[:,0],boundingboxes[:,3]-boundingboxes[:,1],masic,img.shape[1],img.shape[0])
                for y in range(boundingboxes.shape[0]):
            #img = drawBoxes(img, boundingboxes)
            
            #for i in range(5):
            #  cv2.circle(img,(points[y,i],points[y,i+5]),2,(0,255,0))
            #if masic_type==0:
            #   cv2.line(img,(points[y,0],points[y,5]),(points[y,1],points[y,6]),(0,0,0),thickness=20)
            #else:
                
                    img[masic[y][1]:masic[y][3],masic[y][0]:masic[y][2]]=0
            tracknum+=1
        else:
            tracknum+=1
            boundingboxes,points=track_face(img,minsize,RNet,ONet,threshold2,False,factor,boundingboxes)
            #img = drawBoxes(img, boundingboxes)
            if boundingboxes.shape[0]>0:
                masic=createmasic(points,boundingboxes[:,2]-boundingboxes[:,0],boundingboxes[:,3]-boundingboxes[:,1],masic,img.shape[1],img.shape[0])
                for y in range(boundingboxes.shape[0]):
            
            
            #for i in range(5):
            #  cv2.circle(img,(points[y,i],points[y,i+5]),2,(0,255,0))
            #if masic_type==0:
            #   cv2.line(img,(points[y,0],points[y,5]),(points[y,1],points[y,6]),(0,0,0),thickness=20)
            #else:
                    img[masic[y][1]:masic[y][3],masic[y][0]:masic[y][2]]=0
            
        if tracknum>trackspeed:
            tracknum=0
        if boundingboxes.shape[0]==0:
            trackspeed=3
        else:
            trackspeed=orispeed
        videoWriter.write(img)
        #print "points:",points
        cv2.imshow('img', img)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break
        success, img = video.read() 
        whole_end=cv2.getTickCount()
        print("network time:",(network_end-network_begin)/tickfre,"whole time:",(whole_end-whole_begin)/tickfre)
        
    
    videoWriter.release()
    video.release()
    cv2.destroyAllWindows()
    return 0
if __name__ == "__main__":
    main()
    print("finish")
