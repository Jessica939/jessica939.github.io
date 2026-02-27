# leetcode hot 100刷题笔记
## 1.两数之和
为了使时间复杂度小于$O(n^2)$，使用哈希表，此时时间复杂度仅为$O(n)$
```c++
#include <unordered_map>
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;
        for (int i=0; i<nums.size(); i++){
            int complement=target-nums[i];
            if (hashtable.find(complement)!=hashtable.end()){
                return {hashtable[complement], i};
            }
            else hashtable[nums[i]]=i;
        }
        return {};
    }
};
```
## 2.字母异位词分组
使用map，以sort后的字符作为map的索引，然后vector里存字母异位词
```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (const string& s : strs){
            string key = s;
            sort(key.begin(), key.end());
            mp[key].push_back(s);
        }
        vector<vector<string>> ans;
        for (auto& a:mp){
            ans.push_back(a.second);
        }
        return ans;
    }
};
```
## 3.最长连续序列