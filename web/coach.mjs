import rules from './rules.mjs';

export const exercises = [
  {id:'squats', name:'Squats', group:'Lower body', view:'Side', ready:'Stand tall to start', tip:'Stand sideways and keep your hips, knees, and feet visible.'},
  {id:'pushups', name:'Push-ups', group:'Upper body', view:'Side', ready:'Start in a high plank', tip:'Set the camera low and sideways so it can see you from shoulders to feet.'},
  {id:'pullups', name:'Pull-ups', group:'Upper body', view:'Front', ready:'Start with arms extended', tip:'Face the camera and frame your hands, upper body, and feet.'},
  {id:'situp', name:'Sit-ups', group:'Core', view:'Side', ready:'Start lying down', tip:'Set the camera low and sideways so your shoulders, knees, and feet are visible.'},
  {id:'jumping_jacks', name:'Jumping jacks', group:'Full body', view:'Front', ready:'Start with feet together', tip:'Face the camera with enough room for your hands overhead and feet out wide.'},
];

const cueText={
  'shallow depth':'Lower your hips a little farther',
  'too deep':'Stop a little higher',
  'too upright, lean forward slightly':'Lean forward slightly',
  'excessive forward lean':'Keep your chest lifted',
  'knees too far past toes':'Try sitting your hips back slightly',
  'knees caving inward':'Keep knees in line with your toes',
  'arms not fully extended':'Extend your arms at the top',
  'not low enough':'Lower your chest a little farther',
  'hips sagging':'Lift your hips into line',
  'hips piked':'Lower your hips into line',
  'arms not fully extended at bottom':'Extend your arms at the bottom',
  'chin not over the bar':'Bring your chin higher',
  'swinging or kipping':'Keep your lower body still',
  'not sitting up to vertical':'Sit up a little taller',
  'not lowering all the way':'Lower your shoulders to the floor',
  'legs too straight':'Keep your knees bent',
  'feet not wide enough':'Land with your feet a little wider',
  'arms not fully overhead':'Reach your arms overhead',
  'feet not returning together':'Bring your feet back together',
  'arms not returning to sides':'Return your arms to your sides'
};
const joint={shoulder:11,elbow:13,wrist:15,hip:23,knee:25,ankle:27,mouth:9};
const diff=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const dist=(a,b)=>Math.hypot(...diff(a,b));
const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
const angle=(a,b,c)=>{
  const x=diff(a,b),y=diff(c,b),n=Math.hypot(...x)*Math.hypot(...y);
  return n>1e-8?Math.acos(Math.max(-1,Math.min(1,(x[0]*y[0]+x[1]*y[1])/n)))*180/Math.PI:NaN;
};
const tilt=(a,b)=>Math.atan2(Math.abs(a[0]-b[0]),a[1]-b[1])*180/Math.PI;
const arm=b=>angle(b('shoulder'),b('elbow'),b('wrist'));
const leg=b=>angle(b('hip'),b('knee'),b('ankle'));
const raise=b=>angle(b('elbow'),b('shoulder'),b('hip'));
const thigh=b=>tilt(b('knee'),b('hip'));
const torso=b=>tilt(b('hip'),b('shoulder'));
const rise=b=>{const d=diff(b('shoulder'),b('hip'));return Math.atan2(-d[1],-d[0]*(b('ankle')[0]>=b('hip')[0]?1:-1))*180/Math.PI;};
const distBoth=(b,name)=>dist(b(name,0),b(name,1));
const midpoint=(b,name)=>mid(b(name,0),b(name,1));
const width=b=>distBoth(b,'ankle')/distBoth(b,'shoulder');
const chin=b=>(midpoint(b,'mouth')[1]-midpoint(b,'wrist')[1])/dist(midpoint(b,'shoulder'),midpoint(b,'hip'));
const cave=b=>{
 const h=b('hip'), k=b('knee'), a=b('ankle'), axis=diff(a,h);
 const cross=p=>{const d=diff(p,h);return axis[0]*d[1]-axis[1]*d[0];};
 return (180-angle(h,k,a))*(cross(k)*cross(midpoint(b,'hip'))>0?1:-1);
};
const hipIssue=b=>{
 const s=b('shoulder'),h=b('hip'),a=b('ankle');
 const y=s[1]+(h[0]-s[0])*(a[1]-s[1])/(a[0]-s[0]+1e-9);
 return h[1]>y?'hips sagging':'hips piked';
};
const measures={
 S2:thigh,S3:torso,S4:b=>tilt(b('ankle'),b('knee')),S5:cave,
 P1:arm,P2:arm,P3:b=>angle(b('shoulder'),b('hip'),b('ankle')),
 U1:arm,U2:chin,U3:torso,T1:rise,T2:rise,T3:leg,J1:width,J2:raise,J3:width,J4:raise
};
const visible=(p,i)=>p[i]&&p[i][2]>=.6&&p[i][0]>=0&&p[i][0]<=1&&p[i][1]>=0;
const seen=(p,keys,side)=>keys.every(k=>visible(p,joint[k]+side));
const nearSide=(p,keys)=>{
 const score=s=>keys.reduce((sum,k)=>sum+(p[joint[k]+s]?.[2]||0),0);
 return score(1)>score(0)?1:0;
};
const body=(p,side,used=[])=> (k,s=side)=>{const i=joint[k]+s;used.push(i);return p[i];};

export function assess(exercise,pose,p,baseline){
 const viewBody=body(p,0);
 const view=distBoth(viewBody,'shoulder')/dist(midpoint(viewBody,'shoulder'),midpoint(viewBody,'hip'))>=.4?'front':'side';
 const sides=view==='front'?[0,1]:[nearSide(p,['shoulder','hip','knee','ankle','elbow','wrist'])];
 return rules.filter(r=>r.exercise===exercise&&r.pose.includes(pose)&&r.view.includes(view)).flatMap(r=>{
   const results=sides.flatMap(s=>{
     const used=[], b=body(p,s,used);
     let val=measures[r.id](b);
     if(r.id==='S5'){
       if(!baseline)return [];
       const baseUsed=[];
       val-=measures.S5(body(baseline,s,baseUsed));
       if(!baseUsed.every(i=>visible(baseline,i)))return [];
     }
     if(!used.every(i=>visible(p,i))||!Number.isFinite(val))return [];
     let [issue,past]=r.lo!==null&&val<r.lo?[r.low,r.lo-val]:r.hi!==null&&val>r.hi?[r.high,val-r.hi]:['',0];
     if(r.id==='P3'&&issue)issue=hipIssue(b);
     const severity=issue?Math.min(1,past/(r.unit==='deg'?30:.5)):0;
     return [{id:r.id,label:r.label,part:r.part,issue,text:cueText[issue],severity}];
   }).sort((a,b)=>b.severity-a.severity);
   return results.slice(0,1);
 });
}

const motion={
 squats:{work:p=>180-leg(body(p,nearSide(p,['hip','knee','ankle']))),keys:['hip','knee','ankle'],rest:20,move:48,depth:p=>thigh(body(p,nearSide(p,['hip','knee'])))},
 pushups:{work:p=>180-arm(body(p,nearSide(p,['shoulder','elbow','wrist']))),keys:['shoulder','elbow','wrist'],rest:34,move:54},
 pullups:{work:p=>180-arm(body(p,nearSide(p,['shoulder','elbow','wrist']))),keys:['shoulder','elbow','wrist'],rest:34,move:60},
 situp:{work:p=>rise(body(p,nearSide(p,['shoulder','hip','ankle']))),keys:['shoulder','hip','ankle'],rest:31,move:55},
 jumping_jacks:{work:p=>(raise(body(p,0))+raise(body(p,1)))/2,keys:['shoulder','elbow','hip'],rest:49,move:106,both:true}
};

export class RepCoach{
 constructor(id){this.id=id;this.config=motion[id];this.phase='waiting';this.samples=[];this.last=-Infinity;this.reps=0;this.start=0;this.peak=null;this.baseline=null;}
 update(t,landmarks,width,height){
   // Keep x in image-width units so pixel-space angles match the Python checks.
   const p=landmarks?.map(l=>[l.x,l.y*height/width,(l.y >= 0 && l.y <= 1) ? (l.visibility ?? 1) : 0]);
   const fit=key=>[0,1].map(s=>seen(p,key,s)&&key.every(k=>p[joint[k]+s][1]<=height/width));
   if(!p||!fit(['shoulder','hip','knee','ankle']).some(Boolean)||!fit(this.config.keys)[this.config.both?0:nearSide(p,this.config.keys)]||(this.config.both&&!fit(this.config.keys)[1]))return {seen:false,phase:this.phase};
   if(t-this.last>.5){this.phase='waiting';this.samples=[];this.peak=null;this.baseline=null;}
   this.last=t;
   const v=this.config.work(p);
   if(!Number.isFinite(v))return {seen:false,phase:this.phase};
   this.samples.push(v);if(this.samples.length>5)this.samples.shift();
   if(this.samples.length<3)return {seen:true,phase:this.phase};
   const sorted=[...this.samples].sort((a,b)=>a-b), sm=sorted[Math.floor(sorted.length/2)];
   let result;
   if(sm<=this.config.rest){
     if(this.phase==='work'&&t-this.start>=.3&&this.peak){
       const restPose=['squats','pushups'].includes(this.id)?'up':'down';
       const checks=[...assess(this.id,restPose==='up'?'down':'up',this.peak.p,this.baseline),...assess(this.id,restPose,p,this.baseline)];
       const byRule = new Map();
       for (const check of checks) {
         if (check.severity >= 1/3 && check.severity > (byRule.get(check.id)?.severity ?? 0)) {
           byRule.set(check.id, check);
         }
       }
       const issues = [...byRule.values()].sort((a,b)=>b.severity-a.severity);
       result={rep:++this.reps,checks:checks.length,issues};
     }
     this.phase='rest';this.start=t;this.peak=null;this.baseline=p;
   }else if(this.phase==='rest'&&sm>=this.config.move){
     this.phase='work';
   }
   if(this.phase!=='waiting'&&sm>this.config.rest){
     const peak=this.config.depth?this.config.depth(p):v;
     if(!this.peak||peak>this.peak.value)this.peak={p,value:peak};
   }
   return {seen:true,phase:this.phase,result};
 }
}

export const skeleton=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32],[0,11],[0,12]];
